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
from typing import Tuple, Union, Any

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.nn.parallel import DataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from tqdm import tqdm

from src.news_seg.models.dh_segment import DhSegment
from src.news_seg.models.dh_segment_cbam import DhSegmentCBAM
from src.news_seg.models.dh_segment_small import DhSegmentSmall
from src.news_seg.models.trans_unet import VisionTransformer
from src.news_seg.news_dataset import NewsDataset
from src.news_seg.preprocessing import CROP_FACTOR, CROP_SIZE, SCALE, Preprocessing
from src.news_seg.utils import multi_class_csi

EPOCHS = 1
DATALOADER_WORKER = 1
IN_CHANNELS, OUT_CHANNELS = 3, 10
VAL_NUMBER = 3

BATCH_SIZE = 32
LEARNING_RATE = 1e-5  # 1e-5 from Paper .001 Standard 0,0001 seems to work well
WEIGHT_DECAY = 1e-6  # 1e-6 from Paper
LOSS_WEIGHTS: torch.Tensor = torch.tensor([
    2.0,
    10.0,
    10.0,
    10.0,
    4.0,
    20.0,
    10.0,
    10.0,
    10.0,
    20.0,
])


# set random seed for reproducibility
# torch.manual_seed(42)

def init_model(load: Union[str, None], device: str, model_str: str, freeze: bool = True) -> Any:
    """
    Initialise model
    :param args:
    :param load: contains path to load the model from. If False, the model will be initialised randomly
    :return: loaded model
    """
    if model_str == "dh_segment":
        # create model
        model: Any = DhSegment(
            [3, 4, 6, 4],
            in_channels=IN_CHANNELS,
            out_channel=OUT_CHANNELS,
            load_resnet_weights=True,
        )
        model = setup_dh_segment(device, load, model, freeze)
    elif model_str == "trans_unet":
        load_backbone = not load
        model = VisionTransformer(
            load_backbone=load_backbone,
            in_channels=IN_CHANNELS,
            out_channel=OUT_CHANNELS,
            load_resnet_weights=True,
        )

        model = model.float()
        if freeze:
            model.encoder.freeze_encoder()
        # load model if argument is None, it does nothing
        model.load(load, device)

        model.encoder.means = torch.tensor((0.485, 0.456, 0.406))
        model.encoder.stds = torch.tensor((0.229, 0.224, 0.225))
    elif model_str == "dh_segment_cbam":
        model = DhSegmentCBAM(
            in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True
        )
        model = setup_dh_segment(device, load, model, freeze)
    elif model_str == "dh_segment_small":
        model = DhSegmentSmall(
            in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True
        )
        model = setup_dh_segment(device, load, model, freeze)
    assert model, "No valid model string supplied in model parameter"
    return model


def setup_dh_segment(
        device: str, load: Union[str, None], model: Any, freeze: bool
) -> Any:
    """
    Setup function for dh_segment and dh_segment_cbam
    :param device:
    :param load: contains path to load the model from. If False, the model will be initialised randomly
    :param model:
    :return:
    """
    model = model.float()
    if freeze:
        model.freeze_encoder()
    # load model if argument is None, it does nothing
    model.load(load, device)
    # set mean and std in a model for normalization
    model.means = torch.tensor((0.485, 0.456, 0.406))
    model.stds = torch.tensor((0.229, 0.224, 0.225))
    return model


def load_score(load: Union[str, None], args: argparse.Namespace) -> Tuple[float, int, int]:
    """
    Load the score corresponding to the loaded model if requestet, as well as the step value to continue logging.
    """
    best_score = 1000
    step = 0
    epoch = 1
    if args.load_score and load:
        with open(f"scores/model_{args.load}.json", "r", encoding="utf-8") as file:
            best_score, step, epoch = json.load(file)
    return best_score, step, epoch


def focal_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0) -> torch.Tensor:
    """
    Calculate softmax focal loss. Migrated from https://github.com/google-deepmind/optax/pull/573/files
    :param prediction: Unnormalized log probabilities, with shape `[batch, num_classes, ...]`.
    :param target: Valid probability distributions (non-negative, sum to 1), e.g a
      one hot encoding specifying the correct class for each input
    :param gamma: Focusing parameter `>=0`. It controls the contribution of higher confidence predictions.
      Defaults to 2.
    :return: loss tensor [batches, ...]"""
    probalbilites = torch.nn.functional.softmax(prediction, dim=1)
    focus = torch.pow(1 - probalbilites, gamma)
    loss = 1 - target * focus * probalbilites
    return torch.sum(loss, dim=1) / OUT_CHANNELS  # type: ignore


def calculate_scores(data: torch.Tensor) -> Tuple[float, float, Tensor]:
    """
    Applies argmax on the channel dimension of the prediction to obtain int class values.
    Then computes jaccard, accuracy and multi class csi score (critical sucess index)
    :param data: Combined and flattened prediction and Target with shape [P, C] P being number of pixels.
    The last Channel contains target data.
    """
    while True:
        pred = data[:, : -1]
        targets = torch.squeeze(data[:, -1].to(torch.uint8))

        jaccard_fun = JaccardIndex(task="multiclass", num_classes=OUT_CHANNELS).to(pred.get_device())
        accuracy_fun = MulticlassAccuracy(num_classes=OUT_CHANNELS, average="weighted").to(pred.get_device())
        confusion_metric = MulticlassConfusionMatrix(num_classes=OUT_CHANNELS).to(pred.get_device())

        pred = torch.argmax(pred, dim=1).type(torch.uint8)

        # pylint: disable=not-callable
        jaccard = jaccard_fun(pred.flatten(), targets.flatten()).item()
        accuracy = accuracy_fun(pred.flatten(), targets.flatten()).item()
        batch_class_acc = multi_class_csi(pred, targets, confusion_metric)

        return jaccard, accuracy, batch_class_acc


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

        # check for cuda
        self.device = args.cuda_device if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.model = DataParallel(init_model(load, self.device, args.model, args.freeze))

        # set optimizer and loss_fn
        self.optimizer = AdamW(
            self.model.parameters(), lr=learningrate, weight_decay=weight_decay
        )  # weight_decay=1e-4
        self.scaler = torch.cuda.amp.GradScaler()

        # load data
        preprocessing = Preprocessing(
            scale=args.scale,
            crop_factor=args.crop_factor,
            crop_size=args.crop_size,
            pad=args.pad,
            reduce_classes=args.reduce_classes
        )
        dataset = NewsDataset(
            preprocessing,
            image_path=f"{args.data_path}images/",
            target_path=f"{args.data_path}targets/",
            limit=args.limit,
            dataset=args.dataset,
        )

        train_set, validation_set, test_set = dataset.random_split((0.9, 0.05, 0.05))
        print(f"train size: {len(train_set)}, test size: {len(validation_set)}")

        # Turn of augmentations on Validation-set
        validation_set.augmentations = False
        test_set.augmentations = False

        # init dataloader
        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
            pin_memory=True,

        )
        self.val_loader = DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
        )

        assert (
                len(self.train_loader) > 0
                and len(self.val_loader) > 0
                and len(self.test_loader) > 0
        ), "At least one Dataset is to small to assemble at least one batch"

        self.cross_entropy = CrossEntropyLoss(weight=LOSS_WEIGHTS)

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
                    unit="batche(s)",
            ) as pbar:
                for images, targets in self.train_loader:
                    start = time()
                    print(f"Batch Start takes:{start - end}")

                    self.optimizer.zero_grad(set_to_none=True)

                    with torch.autocast(self.device):
                        preds = self.model(images.to(self.device))
                        pred = time()
                        print(f"prediction takes:{pred - start}")
                        loss = self.apply_loss(preds, targets.to(self.device))
                    loss_time = time()
                    print(f"loss takes:{loss_time - pred}")

                    # Backpropagation
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    end = time()
                    print(f"backwards step takes:{end - pred}")

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

                    # delete data from gpu cache
                    del images, loss, targets, preds
                    torch.cuda.empty_cache()

                    if self.step % (len(self.train_loader) // VAL_NUMBER) == 0:
                        val_loss, acc, jac = self.validation()

                        # early stopping
                        score: float = val_loss + (1 - acc) + (1 - jac)  # type: ignore
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
            return focal_loss(preds, self.one_hot_encoding(targets).float()).mean()

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

    def validation(self, test_validation: bool = False) -> Tuple[float, float, float]:
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

        loss, jaccard, accuracy, class_acc, class_sum = (
            0.0,
            0.0,
            0.0,
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
            with torch.autocast(self.device):
                pred = self.model(images.to(self.device))
                end = time()
                print(f"Val prediction takes:{end - start}")
                batch_loss = self.apply_loss(pred, targets.to(self.device)).item()
            loss_time = time()
            print(f"Val loss takes:{loss_time - end}")

            accuracy, class_acc, class_sum, jaccard = self.evaluate_batch(accuracy, class_acc,
                                                                          class_sum, jaccard, pred, targets)
            loss += batch_loss
            scores = time()
            print(f"Val scores take:{scores - loss_time}")

            del images, targets, pred, batch_loss
            torch.cuda.empty_cache()

        loss = loss / size
        accuracy = accuracy / (size * self.num_scores_splits)
        jaccard = jaccard / (size * self.num_scores_splits)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.val_logging(
                loss,
                accuracy,
                jaccard,
                (class_acc / class_sum).detach().cpu().numpy(),
                test_validation,
            )

        self.model.train()
        logging = time()
        print(f"Val logging takes:{logging - scores}")

        return loss, accuracy, jaccard

    def evaluate_batch(self, accuracy: float, class_acc: Tensor, class_sum: Tensor,
                       jaccard: float, pred: Tensor, targets: Tensor) -> Tuple[
        float, Tensor, Tensor, float]:
        """
        Evaluates prediction results of one validation(or test) batch. Updates running score variables. Uses Multi
        Threading to speed up score calculation. Despite threading inside python not being able to run on more than
        one process, in this case that works. The reason being, that all scores are calculated using torch functions,
        which bypass GIL by running C in the background. Using python or torch multiprocessing will not work if the
        data being handled is too large.
        :param accuracy: running score variable
        :param batch_loss: running score variable
        :param class_acc: running score variable
        :param class_sum: running score variable
        :param jaccard: running score variable
        :param loss: running score variable
        :param pred: prediction tensor [B,C,H,W]
        :param targets: target tensor [B,H,W]
        :return: Updated result score values
        """
        batch_class_acc = torch.zeros(OUT_CHANNELS)
        batch_class_sum = torch.zeros(OUT_CHANNELS)

        pred = torch.nn.functional.softmax(pred, dim=1)
        targets = targets.to(self.device)

        pred_batches = self.split_batches(torch.cat((pred, targets[:, None, :, :]), dim=1), (0, 2, 3, 1))

        with ThreadPool(self.num_processes) as pool:
            results = pool.map(calculate_scores, pred_batches)

        for i in range(self.num_scores_splits):
            result = results[i]
            jaccard += result[0]
            accuracy += result[1]
            batch_class_acc += torch.nan_to_num(result[2].detach().cpu())
            batch_class_sum += 1 - torch.isnan(
                batch_class_acc).int()  # ignore pylint error. This comparison detects nan values

        class_acc += batch_class_acc
        class_sum += batch_class_sum

        return accuracy, class_acc, class_sum, jaccard

    def split_batches(self, tensor: torch.Tensor, permutation: Tuple[int, ...]) -> torch.Tensor:
        """
        Splits tensor into self.num_scores_splits chunks. This is necessary to not overload the multiprocessing Queue.
        :param permutation: permutation for this tensor. On a tensor with feature dimensions,
        the feature dimension should be transferred to the end. Everything else has to stay in the same order.
        :param tensor: [B,C,H,W]
        :return: ndarray version of result
        """
        tensor = torch.permute(tensor, permutation).flatten(0, 2)
        return torch.stack(torch.split(tensor, tensor.shape[0] // self.num_scores_splits))  # type: ignore

    def val_logging(
            self,
            loss: float,
            jaccard: float,
            accuracy: float,
            class_accs: ndarray,
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
            if not np.isnan(acc):
                self.summary_writer.add_scalar(
                    f"multi-acc-{environment}/class {i}", acc, global_step=self.step
                )

        self.summary_writer.add_image(
            f"image/{environment}-input",
            torch.squeeze(image.float().cpu()),
            global_step=self.step,
        )  # type:ignore
        self.summary_writer.add_image(
            f"image/{environment}-target",
            target.float().cpu()[
            None,
            :,
            :,
            ]
            / OUT_CHANNELS,
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
        torch.cuda.empty_cache()


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="train")
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
        "--limit",
        type=int,
        default=None,
        help="limit quantity of loaded images for testing purposes",
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
        default="transcribus",
        help="which dataset to expect. Options are 'transcribus' and 'HLNA2013' "
             "(europeaner newspaper project)",
    )
    parser.add_argument(
        "--load-score",
        "-ls",
        action="store_true",
        help="Whether the score corresponding to the loaded model should be loaded as well.",
    )
    parser.add_argument(
        "--no-freeze",
        dest="freeze",
        action="store_false",
        help="Deactivate encoder freezing",
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
        default=2,
        help="Number of batches that will be loaded by each worker in advance.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="dh_segment",
        help="which model to load options are 'dh_segment, trans_unet, dh_segment_small",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="cross_entropy",
        help="Which loss function to use. Available are [cross_entropy, focal_loss]",
    )
    parser.add_argument(
        "--pad",
        "-p",
        type=int,
        nargs="+",
        default=None,
        help="Size to which the image will be padded to. Has to be a tuple (W, H). "
             "Has to be grater or equal to actual image after scaling",
    )
    parser.add_argument(
        "--reduce-classes",
        "-r",
        action="store_true",
        help="If activated, classes are merged into 3 categories. Those being Text, normal "
             "separators and big separators.",
    )

    return parser.parse_args()


def main() -> None:
    """Main method creates directories if not already present, sets up Trainer and handels top level logging.
    """
    parameter_args = get_args()

    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("scores"):
        os.makedirs("scores")
    if not os.path.exists("logs/worker-experiment"):
        os.makedirs("logs/worker-experiment")

    torch.manual_seed(parameter_args.torch_seed)

    # setup tensor board
    train_log_dir = "logs/runs/" + parameter_args.name
    summary_writer = SummaryWriter(train_log_dir, max_queue=1000, flush_secs=3600)

    load_model = f"models/model_{parameter_args.load}.pt" if parameter_args.load else None

    trainer = Trainer(
        load=load_model,
        save_model=f"models/model_{parameter_args.name}",
        save_score=f"scores/model_{parameter_args.name}",
        batch_size=parameter_args.batch_size,
        learningrate=parameter_args.lr,
        weight_decay=parameter_args.wd,
        summary=summary_writer,
        args=parameter_args
    )

    print(f"Training run {parameter_args.name}")
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

    duration = trainer.train(epochs=parameter_args.epochs)
    with open(f"logs/worker-experiment/{parameter_args.num_workers}_{parameter_args.prefetch_factor}.json", "w",
              encoding="utf-8") as file:
        json.dump((parameter_args.num_workers, parameter_args.prefetch_factor, duration), file)


if __name__ == "__main__":
    main()
