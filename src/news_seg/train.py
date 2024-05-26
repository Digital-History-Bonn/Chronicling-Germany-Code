"""
module for training the hdSegment Model
"""

import argparse
import datetime
import json
import os
import warnings
from multiprocessing.pool import ThreadPool
from time import time
from typing import Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.nn.parallel import DataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

from src.news_seg.train_config import BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, LOSS_WEIGHTS, VAL_NUMBER, OUT_CHANNELS, \
    EPOCHS, DATALOADER_WORKER, DEFAULT_SPLIT
from src.news_seg.helper.train_helper import init_model, load_score, focal_loss, calculate_scores, initiate_datasets, \
    initiate_dataloader
from src.news_seg.utils import split_batches, adjust_path
from src.news_seg.processing.preprocessing import CROP_FACTOR, CROP_SIZE, SCALE
from src.news_seg.helper.train_helper import multi_precison_recall


class Trainer:
    """Training class containing functions for training and validation."""

    def __init__(
            self,
            args: argparse.Namespace,
            save_model: str,
            save_score: str,
            summary: SummaryWriter,
            load: Union[str, None] = None,
            batch_size: int = BATCH_SIZE,
            learningrate: float = LEARNING_RATE,
            weight_decay: float = WEIGHT_DECAY
    ):
        """
        Trainer-class to train DhSegment Model
        :param load: model to load, init random if None
        :param save_model: path to the model savefile
        :param save_score: path to the score savefile
        :param batch_size: size of batches
        :param learningrate: learning-rate
        """

        # init params
        self.summary_writer = summary
        batch_size = args.gpu_count * batch_size
        self.best_score, self.step, self.epoch = load_score(load, args)
        self.save_model = save_model
        self.save_score = save_score
        self.learningrate: float = learningrate
        self.batch_size: int = batch_size
        self.best_step = 0
        self.loss = args.loss
        self.num_processes = args.num_processes
        self.num_scores_splits = args.num_scores_splits
        self.amp = args.amp
        self.clip = args.clip
        self.scheduler_type = args.scheduler

        # check for cuda
        self.device = args.cuda_device if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.model = DataParallel(
            init_model(load, self.device, args.model, args.freeze, args.skip_cbam, args.override_load_channels))

        # set optimizer and loss_fn
        self.optimizer = AdamW(
            self.model.parameters(), lr=learningrate, weight_decay=weight_decay
        )  # weight_decay=1e-4
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        if args.scheduler == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', 0.5, 15)
        elif args.scheduler == 'cosine_annealing':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, 5, 1, eta_min=0, last_epoch=-1)  # type: ignore

        self.cross_entropy = CrossEntropyLoss(weight=LOSS_WEIGHTS)

        # load data
        train_set, validation_set, test_set = initiate_datasets(args)

        self.train_loader, self.val_loader, self.test_loader = initiate_dataloader(args, batch_size, test_set,
                                                                                   train_set, validation_set)

    def train(self, epochs: int = 1) -> float:
        """
        executes all training epochs. After each epoch a validation round is performed.
        :param epochs: number of epochs that will be executed
        :return: None
        """
        training_start = time()
        self.model.to(self.device)
        self.cross_entropy.to(self.device)
        end = time()

        for self.epoch in range(self.epoch, epochs + 1):
            self.model.train()

            with tqdm(
                    total=(len(self.train_loader)),
                    desc=f"Epoch {self.epoch}/{epochs}",
                    unit="batch(es)",
            ) as pbar:
                for images, targets in self.train_loader:
                    # transfer = time()
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    # transfer_end = time()
                    # print(f"Transfer takes:{transfer_end - transfer}")

                    start = time()
                    print(f"Batch Start takes:{start - end}")

                    with torch.autocast(self.device, enabled=self.amp):
                        preds = self.model(images.to(self.device, non_blocking=True))
                        # pred = time()
                        # print(f"prediction takes:{pred - start}")
                        loss = self.apply_loss(preds, targets.to(self.device, non_blocking=True))
                    # loss_time = time()
                    # print(f"loss takes:{loss_time - pred}")

                    # Backpropagation
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.optimizer.zero_grad(set_to_none=True)

                    backward = time()
                    print(f"backwards step takes:{backward - start}")

                    # update tensor board logs
                    self.step += 1
                    # pylint: disable-next=not-context-manager

                    self.summary_writer.add_scalar(
                        "train loss", loss.item(), global_step=self.step
                    )  # type:ignore
                    # self.summary_writer.add_scalar('batch mean', images.detach().cpu().mean(),
                    # global_step=self.step) #type:ignore
                    # self.summary_writer.add_scalar('batch std', images.detach().cpu().std(),
                    # global_step=self.step) #type:ignore
                    # self.summary_writer.add_scalar('target batch mean', targets.detach().cpu().float().mean(),
                    # global_step=self.step) #type:ignore
                    # self.summary_writer.add_scalar('target batch std', targets.detach().cpu().float().std(),
                    # global_step=self.step) #type:ignore

                    # update description
                    pbar.update(1)
                    pbar.set_postfix(**{"loss (batch)": loss.item()})

                    # cache = time()
                    # delete data from gpu cache
                    del images, loss, targets, preds
                    # torch.cuda.empty_cache()
                    # cache_end = time()
                    # print(f"cache takes:{cache_end - cache}")

                    if self.step % (len(self.train_loader) // VAL_NUMBER) == 0:
                        _, _, _, class_acc = self.validation()

                        # early stopping
                        score: float = (1 - np.mean(np.nan_to_num(class_acc)))  # type: ignore

                        if self.scheduler_type:
                            if self.scheduler_type == "reduced_on_plateau":
                                self.scheduler.step(score)
                            else:
                                self.scheduler.step(self.epoch)
                            self.summary_writer.add_scalar(
                                "lr", self.scheduler.get_last_lr(), global_step=self.step
                            )  # type:ignore

                        if score < self.best_score:
                            # update cur_best value
                            self.best_score = score
                            self.best_step = self.step
                            print(
                                f"saved model because of early stopping with value {score}"
                            )

                            self.model.module.save(self.save_model + "_best")  # type: ignore

                    # log the step of current best model
                    # pylint: disable-next=not-context-manager
                    self.summary_writer.add_scalar(
                        "current best", self.best_step, global_step=self.step
                    )  # type:ignore
                    end = time()
                    print(f"rest takes:{end - backward}")

            # save model at end of epoch
            self.model.module.save(self.save_model)  # type: ignore
            with open(f"{self.save_score}.json", "w", encoding="utf-8") as file:
                json.dump((score, self.step, self.epoch + 1), file)
            self.summary_writer.flush()

        self.validation(test_validation=True)
        training_end = time()
        return round(training_end - training_start, ndigits=2)

    def apply_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Applies configured loss function
        :param preds: prdictions tensor with shape[B,C,H,W]
        :param targets: target tensor with shape [H,W]
        :return: loss scalar
        """
        if self.loss == "focal_loss":
            return focal_loss(preds, self.one_hot_encoding(targets).float(), LOSS_WEIGHTS.to(self.device)).mean()

        return self.cross_entropy(preds, targets)  # type: ignore

    def one_hot_encoding(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Handels one hot encoding of target to be usable for loss function.s
        :param targets: 2d target with label values
        :return: 3d target with one channel for each class
        """
        # pylint: disable-next=not-callable
        return torch.permute(one_hot(targets.to(self.device), num_classes=OUT_CHANNELS),
                             (0, 3, 1, 2))

    def validation(self, test_validation: bool = False) -> Tuple[float, float, float, ndarray]:
        """
        Executes one validation round, containing the evaluation of the current model on the entire validation set.
        jaccard score, accuracy and multiclass accuracy are calculated over the validation set. Multiclass accuracy
        also tracks a class sum value, to handle nan values from MulticlassAccuracy
        Is also being used for test-evaluation at the end of training with another dataset.
        :return: None
        """

        loader = self.test_loader if test_validation else self.val_loader
        self.model.eval()
        size = len(loader)

        loss, jaccard, accuracy, class_acc, class_sum, precision, precision_sum, recall, recall_sum = (
            0.0,
            0.0,
            0.0,
            torch.zeros(OUT_CHANNELS),
            torch.zeros(OUT_CHANNELS),
            torch.zeros(OUT_CHANNELS),
            torch.zeros(OUT_CHANNELS),
            torch.zeros(OUT_CHANNELS),
            torch.zeros(OUT_CHANNELS),
        )
        end = time()

        pstart_time = time()
        print(f"process start take:{pstart_time - end}")

        for images, targets in tqdm(
                loader, desc="validation_round", total=size, unit="batch(es)"
        ):
            start = time()
            print(f"Val Start takes:{start - end}")
            with torch.autocast(self.device, enabled=self.amp):
                pred = self.model(images.to(self.device))
                end = time()
                print(f"Val prediction takes:{end - start}")
                batch_loss = self.apply_loss(pred, targets.to(self.device)).item()
            loss_time = time()
            print(f"Val loss takes:{loss_time - end}")

            accuracy, class_acc, class_sum, jaccard, precision, precision_sum, recall, recall_sum = self.evaluate_batch(
                accuracy, class_acc, class_sum, jaccard, pred, targets, test_validation, precision, precision_sum,
                recall, recall_sum)

            loss += batch_loss
            scores = time()
            print(f"Val scores take:{scores - loss_time}")

            del images, targets, pred, batch_loss
            # torch.cuda.empty_cache()

        loss = loss / size
        accuracy = accuracy / (size * self.num_scores_splits)
        jaccard = jaccard / (size * self.num_scores_splits)
        class_acc_ndarray = (class_acc / class_sum).detach().cpu().numpy()
        precision_ndarray = (precision / precision_sum).detach().cpu().numpy()
        recall_ndarray = (recall / recall_sum).detach().cpu().numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.val_logging(
                loss,
                accuracy,
                jaccard,
                class_acc_ndarray,
                precision_ndarray,
                recall_ndarray,
                test_validation,
            )

        self.model.train()
        logging = time()
        print(f"Val logging takes:{logging - scores}")

        return loss, accuracy, jaccard, class_acc_ndarray

    def evaluate_batch(self, accuracy: float, class_acc: Tensor, class_sum: Tensor,
                       jaccard: float, pred: Tensor, targets: Tensor, test_validation: bool, precision: Tensor,
                       precision_sum: Tensor, recall: Tensor, recall_sum: Tensor) -> Tuple[
        float, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor]:
        """
        Evaluates prediction results of one validation(or test) batch. Updates running score variables. Uses Multi
        Threading to speed up score calculation. Despite threading inside python not being able to run on more than
        one process, in this case that works. The reason being, that all scores are calculated using torch functions,
        which bypass GIL by running C in the background. Using python or torch multiprocessing will not work if the
        data being handled is too large.
        :param precision_sum: running sum variable
        :param recall_sum: running sum variable
        :param precision: running score variable
        :param recall: running score variable
        :param accuracy: running score variable
        :param batch_loss: running score variable
        :param class_acc: running score variable
        :param class_sum: running sum variable
        :param jaccard: running score variable
        :param loss: running score variable
        :param pred: prediction tensor [B,C,H,W]
        :param targets: target tensor [B,H,W]
        :param test_validation: Activates precision and recall calculation on test runs.
        :return: Updated result score values
        """
        batch_class_acc = torch.zeros(OUT_CHANNELS)
        batch_class_sum = torch.zeros(OUT_CHANNELS)

        pred = torch.nn.functional.softmax(pred, dim=1)
        targets = targets.to(self.device)

        batch_precision = torch.zeros(OUT_CHANNELS)
        batch_recall = torch.zeros(OUT_CHANNELS)
        if test_validation:
            batch_precision, batch_recall = multi_precison_recall(pred, targets, OUT_CHANNELS)

        pred_batches = split_batches(torch.cat((pred, targets[:, None, :, :]), dim=1), (0, 2, 3, 1),
                                     self.num_scores_splits)

        with ThreadPool(self.num_processes) as pool:
            results = pool.map(calculate_scores, pred_batches)

        for i in range(self.num_scores_splits):
            result = results[i]
            jaccard += result[0]
            accuracy += result[1]
            batch_class_acc += torch.nan_to_num(result[2].detach().cpu())
            batch_class_sum += 1 - torch.isnan(
                result[2].detach().cpu()).int()  # ignore pylint error. This comparison detects nan values

        class_acc += batch_class_acc
        class_sum += batch_class_sum

        precision += torch.nan_to_num(batch_precision.detach().cpu())
        precision_sum += 1 - torch.isnan(
            batch_precision.detach().cpu()).int()

        recall += torch.nan_to_num(batch_recall.detach().cpu())
        recall_sum += 1 - torch.isnan(
            batch_recall.detach().cpu()).int()

        return accuracy, class_acc, class_sum, jaccard, precision, precision_sum, recall, recall_sum

    def val_logging(
            self,
            loss: float,
            jaccard: float,
            accuracy: float,
            class_accs: ndarray,
            precision: ndarray,
            recall: ndarray,
            test_validation: bool,
    ) -> None:
        """Handles logging for loss values and validation images. Per epoch one random cropped image from the
        validation set will be evaluated. Furthermore, one full size image will be predicted and logged.
        :param test_validation: if true the test dataset will be used for validation
        :param loss: loss sum for validation round
        :param jaccard: jaccard score of validation round
        :param accuracy: accuracy of validation round
        :param class_accs: array of accuracy by class
        """
        # select random image and it's target
        loader = self.test_loader if test_validation else self.val_loader
        size = len(loader)
        random_idx = np.random.randint(
            0, (size * loader.batch_size if loader.batch_size else 1)
        )
        image, target = loader.dataset[random_idx]
        image = image[None, :]

        # predict image
        pred = torch.nn.functional.softmax(self.model(image.to(self.device))).argmax(dim=1).float()

        environment = "test" if test_validation else "val"

        # update tensor board logs
        # pylint: disable-next=not-context-manager
        self.summary_writer.add_scalar("epoch", self.epoch, global_step=self.step)

        self.summary_writer.add_scalar(f"{environment}/loss", loss, global_step=self.step)
        self.summary_writer.add_scalar(
            f"{environment}/accuracy", accuracy, global_step=self.step
        )

        self.summary_writer.add_scalar(
            f"{environment}/jaccard score", jaccard, global_step=self.step
        )

        for i, acc in enumerate(class_accs):
            if np.isnan(acc):
                acc = 0
            self.summary_writer.add_scalar(
                f"multi-acc-{environment}/class {i}", acc, global_step=self.step
            )

        if test_validation:
            for i, value in enumerate(precision):
                if np.isnan(value):
                    value = 0
                self.summary_writer.add_scalar(
                    f"multi-precision-{environment}/class {i}", value, global_step=self.step
                )
            for i, value in enumerate(recall):
                if np.isnan(value):
                    value = 0
                self.summary_writer.add_scalar(
                    f"multi-recall-{environment}/class {i}", value, global_step=self.step
                )

        self.summary_writer.add_image(
            f"image/{environment}-input",
            torch.squeeze(image.float().cpu()),
            global_step=self.step,
        )  # type:ignore
        self.summary_writer.add_image(
            f"image/{environment}-target",
            target.float().cpu()[None, :, :, ] / OUT_CHANNELS,
            global_step=self.step,
        )  # type:ignore
        self.summary_writer.add_image(
            f"image/{environment}-prediction",
            pred.float().cpu() / OUT_CHANNELS,
            global_step=self.step,
        )  # type:ignore

        print(f"average loss: {loss}")
        print(f"average accuracy: {accuracy}")
        print(f"average jaccard score: {jaccard}")  # Intersection over Union

        del size, image, target, pred
        # torch.cuda.empty_cache()

    def get_test_score(self, model_path: str) -> Tuple[float, float]:
        """
        After Training has finished, load best model and run test-dataset validation on it.
        :param model_path: name of model to be evaluated.
        :return: Tuple of scores with values from 0 to 1. First ist composed out of accuracy and jaccard score.
        Second is the mean of multi clas csi results.
        """
        self.model.module.load(model_path, self.device)

        self.model.module.means = torch.tensor((0.485, 0.456, 0.406))
        self.model.module.stds = torch.tensor((0.229, 0.224, 0.225))

        self.model.to(self.device)

        self.step += 1

        _, acc, jac, class_acc = self.validation(True)

        score = np.mean(np.array([acc, jac]))
        class_score = np.mean(np.nan_to_num(class_acc))

        return round(float(score), ndigits=4), round(float(class_score), ndigits=4)


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="Train scipt for Newspaper Layout Models.")
    parser.add_argument(
        "--epochs",
        "-e",
        metavar="EPOCHS",
        type=int,
        default=EPOCHS,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--name",
        "-n",
        metavar="NAME",
        type=str,
        default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="name of run in tensorboard",
    )
    parser.add_argument(
        "--id",
        metavar="ID",
        type=str,
        default=None,
        help="Id for experiment runs. Dictates the filename of custom log files.",
    )
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        metavar="LR",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--weight-decay",
        "-wd",
        metavar="WD",
        type=float,
        default=WEIGHT_DECAY,
        help="Weight decay L2 regularization",
        dest="wd",
    )
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        dest="scale",
        default=SCALE,
        help="Downscaling factor of the images",
    )
    parser.add_argument(
        "--load",
        "-l",
        type=str,
        dest="load",
        default=None,
        help="model to load (default is None)",
    )
    parser.add_argument(
        "--cuda-device", "-c", type=str, default="cuda", help="Cuda device string"
    )
    parser.add_argument(
        "--torch-seed", "-ts", type=float, default=314.0, help="Torch seed"
    )
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        dest="data_path",
        default=None,
        help="path for folder with folders 'images' and 'targets'",
    )
    parser.add_argument(
        "--result-path",
        type=str,
        default="default/",
        help="path to folder in which duration log files should be written",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit quantity of loaded images for the train dataset. ",
    )
    parser.add_argument(
        "--crop-size", type=int, default=CROP_SIZE, help="Window size of image cropping"
    )
    parser.add_argument(
        "--crop-factor",
        type=float,
        default=CROP_FACTOR,
        help="Scaling factor for cropping steps",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="transkribus",
        help="which dataset to expect. Options are 'transkribus' and 'HLNA2013' "
             "(europeaner newspaper project)",
    )
    parser.add_argument(
        "--load-score",
        "-ls",
        action="store_true",
        help="Whether the score corresponding to the loaded model should be loaded as well.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        help="Select scheduler [reduce_on_plateau, cosine_annealing].",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Activates automated mixed precision",
    )
    parser.add_argument(
        "--no-freeze",
        dest="freeze",
        action="store_false",
        help="Deactivate encoder freezing",
    )
    parser.add_argument(
        "--no-scale-aug",
        dest="scale_aug",
        action="store_false",
        help="Deactivate scaling augmentation",
    )
    parser.add_argument(
        "--gpu-count",
        "-g",
        type=int,
        default=1,
        help="Number of gpu that should be used for training",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=DATALOADER_WORKER,
        help="Number of workers for the Dataloader",
    )
    parser.add_argument(
        "--num-processes",
        "-np",
        type=int,
        default=4,
        help="Number of processes for the multi process sections within training",
    )
    parser.add_argument(
        "--num-scores-splits",
        "-nss",
        type=int,
        default=128,
        help="Number of chunks that prediction data in validation is split to be processed in parallel",
    )
    parser.add_argument(
        "--prefetch-factor",
        "-pf",
        type=int,
        default=None,
        help="Number of batches that will be loaded by each worker in advance.",
    )
    parser.add_argument(
        "--clip",
        type=int,
        default=10,
        help="Max norm for gradient clipping",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="dh_segment",
        help="which model to load options are 'dh_segment, trans_unet, dh_segment_small",
    )
    parser.add_argument(
        "--skip-cbam",
        action="store_true",
        help="Activates cbam skip connection. Does only have an effect if the cbam dhsegment model is used",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="cross_entropy",
        help="Which loss function to use. Available are [cross_entropy, focal_loss]",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        nargs="+",
        default=DEFAULT_SPLIT,
        help="Takes 3 float values for a custom dataset split ratio. The ratio have to sum up to one and the Dataset "
             "has to be big enough, to contain at least one batch for each dataset. Provide ratios for train, test "
             "and validation in this order.",
    )
    parser.add_argument(
        "--reduce-classes",
        "-r",
        action="store_true",
        help="If activated, classes are merged into 3 categories. Those being Text, normal "
             "separators and big separators.",
    )
    parser.add_argument(
        "--custom-split-file",
        type=str,
        default=None,
        help="Provide path for custom split json file. This should contain a list with file stems "
             "of train, validation and test images. File stems is the file name without the extension.",
    )
    parser.add_argument(
        "--override-load-channels",
        type=int,
        default=OUT_CHANNELS,
        help="This overrides the number of classes, with that a model will be loaded. The pretrained model will be "
             "loaded with this number of output classes instead of the configured number. This is necessary if a "
             "pretrained model is intended to be used for a task with a different number of output classes.",
    )

    return parser.parse_args()


def main() -> None:
    """Main method creates directories if not already present, sets up Trainer and handels top level logging.
    """
    parameter_args = get_args()
    result_path = adjust_path(parameter_args.result_path)

    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("scores"):
        os.makedirs("scores")
    if not os.path.exists(f"logs/{result_path}"):
        os.makedirs(f"logs/{result_path}")

    torch.manual_seed(parameter_args.torch_seed)

    name = f"{parameter_args.name}_{parameter_args.torch_seed}"
    # name = f"{parameter_args.name}"

    # setup tensor board
    train_log_dir = "logs/runs/" + name
    summary_writer = SummaryWriter(train_log_dir, max_queue=1000, flush_secs=3600)

    load_model = f"models/model_{parameter_args.load}.pt" if parameter_args.load else None

    trainer = Trainer(
        load=load_model,
        save_model=f"models/model_{name}",
        save_score=f"scores/model_{name}",
        batch_size=parameter_args.batch_size,
        learningrate=parameter_args.lr,
        weight_decay=parameter_args.wd,
        summary=summary_writer,
        args=parameter_args
    )

    print(f"Training run {name}")
    print(f"Batchsize {parameter_args.batch_size}")
    print(f"LR {parameter_args.lr}")
    print(f"Loss Function {parameter_args.loss}")
    print(f"Weight Decay {parameter_args.wd}")
    print(f"crop-size {parameter_args.crop_size}")
    print(f"model {parameter_args.model}")
    print(f"num-workers {parameter_args.num_workers}")
    print(f"prefetch factor {parameter_args.prefetch_factor}")
    print(f"gpu-count {parameter_args.gpu_count}")
    print(f"num-processes {parameter_args.num_processes}")
    print(f"num-scores-splits {parameter_args.num_scores_splits}")
    print(f"amp:  {parameter_args.amp}")
    print(f"skip cbam: {parameter_args.skip_cbam}")

    duration = trainer.train(epochs=parameter_args.epochs)
    model_path = f"models/model_{name}_best.pt" if trainer.best_step != 0 else \
        f"models/model_{name}.pt"
    score, multi_class_score = trainer.get_test_score(model_path)
    with open(f"logs/{result_path}{name}_{parameter_args.lr}.json",
              "w",
              encoding="utf-8") as file:
        json.dump((parameter_args.batch_size, parameter_args.lr, score, multi_class_score, duration), file)


if __name__ == "__main__":
    main()
