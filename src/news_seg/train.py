"""
module for training the hdSegment Model
"""

import argparse
import datetime
import json
import warnings
from typing import List, Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from sklearn.metrics import accuracy_score, jaccard_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

# from model import DhSegment
# from news_dataset import NewsDataset
# from preprocessing import Preprocessing, CROP_SIZE, CROP_FACTOR
# from preprocessing import SCALE
# from utils import multi_class_csi

from src.news_seg.model import DhSegment
from src.news_seg.news_dataset import NewsDataset
from src.news_seg.preprocessing import Preprocessing, CROP_SIZE, CROP_FACTOR
from src.news_seg.preprocessing import SCALE
from src.news_seg.utils import multi_class_csi

EPOCHS = 1
DATALOADER_WORKER = 1
IN_CHANNELS, OUT_CHANNELS = 3, 10
VAL_NUMBER = 5

BATCH_SIZE = 32
LEARNING_RATE = 1e-5  # 1e-5 from Paper .001 Standard 0,0001 seems to work well
WEIGHT_DECAY = 1e-6  # 1e-6 from Paper
LOSS_WEIGHTS: List[float] = [
    2.0,
    10.0,
    10.0,
    10.0,
    4.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
]  # 1 and 5 seems to work well


# set random seed for reproducibility
# torch.manual_seed(42)


def init_model(load: Union[str, None], device: str) -> DhSegment:
    """
    Initialise model
    :param load: contains path to load the model from. If False, the model will be initialised randomly
    :return: loaded model
    """
    # create model
    model = DhSegment(
        [3, 4, 6, 4],
        in_channels=IN_CHANNELS,
        out_channel=OUT_CHANNELS,
        load_resnet_weights=True,
    )
    model = model.float()
    model.freeze_encoder()
    # load model if argument is None, it does nothing
    model.load(load, device)

    # set mean and std in a model for normalization
    model.means = torch.tensor((0.485, 0.456, 0.406))
    model.stds = torch.tensor((0.229, 0.224, 0.225))
    return model


def load_score(load: Union[str, None]) -> Tuple[float, int, int]:
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


class Trainer:
    """Training class containing functions for training and validation."""

    def __init__(
            self,
            save_model: str,
            save_score: str,
            load: Union[str, None] = None,
            batch_size: int = BATCH_SIZE,
            learningrate: float = LEARNING_RATE,
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
        batch_size = args.gpu_count * batch_size
        self.best_score, self.step, self.epoch = load_score(load)
        self.save_model = save_model
        self.save_score = save_score
        self.learningrate: float = learningrate
        self.batch_size: int = batch_size
        self.best_step = 0

        # check for cuda
        self.device = args.cuda_device if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.model = torch.nn.DataParallel(init_model(load, self.device))

        # set optimizer and loss_fn
        self.optimizer = AdamW(
            self.model.parameters(), lr=learningrate, weight_decay=WEIGHT_DECAY
        )  # weight_decay=1e-4

        # load data
        preprocessing = Preprocessing(scale=args.scale, crop_factor=args.crop_factor, crop_size=args.crop_size)
        dataset = NewsDataset(preprocessing, image_path=f"{args.data_path}images/",
                              target_path=f"{args.data_path}targets/",
                              limit=args.limit, dataset=args.dataset)

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
        )
        self.val_loader = DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
        )

        assert len(self.train_loader) > 0 and len(self.val_loader) > 0 and len(
            self.test_loader) > 0, "At least one Dataset is to small to assemble at least one batch"

        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(LOSS_WEIGHTS).to(self.device)
        )

    def train(self, epochs: int = 1) -> None:
        """
        executes all training epochs. After each epoch a validation round is performed.
        :param epochs: number of epochs that will be executed
        :return: None
        """

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        for self.epoch in range(self.epoch, epochs + 1):
            self.model.train()

            with tqdm(
                total=(len(self.train_loader)),
                desc=f"Epoch {self.epoch}/{epochs}",
                unit="batche(s)",
            ) as pbar:
                for images, targets in self.train_loader:
                    preds = self.model(images.to(self.device))
                    loss = self.loss_fn(preds, targets.to(self.device))

                    # Backpropagation
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                    # update tensor board logs
                    self.step += 1
                    # pylint: disable-next=not-context-manager

                    summary_writer.add_scalar("train loss", loss.item(), global_step=self.step)  # type:ignore
                    # summary_writer.add_scalar('batch mean', images.detach().cpu().mean(),
                    # global_step=self.step) #type:ignore
                    # summary_writer.add_scalar('batch std', images.detach().cpu().std(),
                    # global_step=self.step) #type:ignore
                    # summary_writer.add_scalar('target batch mean', targets.detach().cpu().float().mean(),
                    # global_step=self.step) #type:ignore
                    # summary_writer.add_scalar('target batch std', targets.detach().cpu().float().std(),
                    # global_step=self.step) #type:ignore

                    # update description
                    pbar.update(1)
                    pbar.set_postfix(**{"loss (batch)": loss.item()})

                    # delete data from gpu cache
                    del images, targets, loss, preds
                    torch.cuda.empty_cache()

                    if self.step % (len(self.train_loader) // VAL_NUMBER) == 0:
                        loss, acc = self.validation()

                        # early stopping
                        score = loss + (1 - acc)
                        if score < self.best_score:
                            # update cur_best value
                            self.best_score = loss + (1 - acc)
                            self.best_step = self.step
                            print(
                                f"saved model because of early stopping with value {loss + (1 - acc)}"
                            )

                            self.model.module.save(self.save_model + "_best") # type: ignore

                    # log the step of current best model
                    # pylint: disable-next=not-context-manager
                    summary_writer.add_scalar(
                        "current best", self.best_step, global_step=self.step
                    )  # type:ignore

            # save model at end of epoch
            self.model.module.save(self.save_model) # type: ignore
            with open(f"{self.save_score}.json", "w", encoding="utf-8") as file:
                json.dump((score, self.step, self.epoch + 1), file)
            summary_writer.flush()

        self.validation(test_validation=True)

    def validation(self, test_validation: bool = False) -> Tuple[float, float]:
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
            0,
            0,
            0,
            np.zeros(OUT_CHANNELS),
            np.zeros(OUT_CHANNELS),
        )

        for images, targets in tqdm(
                loader, desc="validation_round", total=size, unit="batch(es)"
        ):
            pred = self.model(images.to(self.device))
            batch_loss = self.loss_fn(pred, targets.to(self.device))

            # detach results
            pred = pred.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            loss += batch_loss.item()

            pred = np.argmax(pred, axis=1)
            jaccard += jaccard_score(targets.flatten(), pred.flatten(), average="macro")
            accuracy += accuracy_score(targets.flatten(), pred.flatten())
            batch_class_acc = multi_class_csi(
                torch.tensor(pred).flatten(), torch.tensor(targets).flatten()
            )
            class_acc += np.nan_to_num(batch_class_acc)
            class_sum += ~np.isnan(
                batch_class_acc
            )  # ignore pylint error. This comparison detects nan values

            del images, targets, pred, batch_loss
            torch.cuda.empty_cache()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.val_logging(
                loss / size,
                jaccard / size,
                accuracy / size,
                class_acc / class_sum,
                test_validation,
            )

        self.model.train()

        return loss / size, accuracy / size

    def val_logging(self, loss: float, jaccard: float, accuracy: float, class_accs: ndarray,
                    test_validation: bool) -> None:
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
        pred = self.model(image.to(self.device)).argmax(dim=1).float()

        environment = "test" if test_validation else "val"

        # update tensor board logs
        # pylint: disable-next=not-context-manager
        summary_writer.add_scalar("epoch", self.epoch, global_step=self.step)

        summary_writer.add_scalar(f"{environment}/loss", loss, global_step=self.step)
        summary_writer.add_scalar(f"{environment}/accuracy", accuracy, global_step=self.step)

        summary_writer.add_scalar(f"{environment}/jaccard score", jaccard, global_step=self.step)

        for i, acc in enumerate(class_accs):
            if not np.isnan(acc):
                summary_writer.add_scalar(
                    f"multi-acc-{environment}/class {i}", acc, global_step=self.step
                )

        summary_writer.add_image(
            f"image/{environment}-input",
            torch.squeeze(image.float().cpu()),
            global_step=self.step,
        )  # type:ignore
        summary_writer.add_image(
            f"image/{environment}-target",
            target.float().cpu()[None, :, :, ] / OUT_CHANNELS,
            global_step=self.step,
        )  # type:ignore
        summary_writer.add_image(
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
    parser.add_argument('--crop_size', type=int, default=CROP_SIZE, help='Window size of image cropping')
    parser.add_argument('--crop_factor', type=float, default=CROP_FACTOR, help='Scaling factor for cropping steps')
    parser.add_argument('--dataset', type=str, default="transcribus",
                        help="which dataset to expect. Options are 'transcribus' and 'HLNA2013' "
                             "(europeaner newspaper project)")
    parser.add_argument(
        "--load-score", "-ls", action='store_true',
        help="Whether the score corresponding to the loaded model should be loaded as well."
    )
    parser.add_argument(
        "--gpu-count", "-g", type=int, default=1, help="Number of gpu that should be used for training"
    )
    parser.add_argument(
        "--num-workers", "-w", type=int, default=DATALOADER_WORKER, help="Number of workers for the Dataloader"
    )


    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.torch_seed)

    # setup tensor board
    train_log_dir = "logs/runs/" + args.name
    summary_writer = SummaryWriter(train_log_dir, max_queue=1000, flush_secs=3600)

    load_model = f"models/model_{args.load}.pt" if args.load else None

    trainer = Trainer(
        load=load_model,
        save_model=f"models/model_{args.name}",
        save_score=f"scores/model_{args.name}",
        batch_size=args.batch_size,
        learningrate=args.lr,
    )

    trainer.train(epochs=args.epochs)
