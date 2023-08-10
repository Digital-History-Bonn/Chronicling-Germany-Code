"""
module for training the hdSegment Model
"""

import argparse
import datetime
import warnings
from typing import List, Union, Tuple

import numpy as np
import tensorflow as tf  # type: ignore
import torch  # type: ignore
from sklearn.metrics import accuracy_score, jaccard_score  # type: ignore
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

from .model import DhSegment
from .news_dataset import NewsDataset
from .preprocessing import SCALE
from .utils import multi_class_csi

EPOCHS = 1
DATALOADER_WORKER = 1
IN_CHANNELS, OUT_CHANNELS = 3, 10
VAL_EVERY = 2500

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

PREDICT_SCALE = 0.25
PREDICT_IMAGE = "../prima/inputs/NoAnnotations/00675238.tif"

# set random seed for reproducibility
# torch.manual_seed(42)


class Trainer:
    """Training class containing functions for training and validation."""

    def __init__(
        self,
        load: Union[str, None] = None,
        save_model: Union[str, None] = None,
        batch_size: int = BATCH_SIZE,
        learningrate: float = LEARNING_RATE,
    ):
        """
        Trainer-class to train DhSegment Model
        :param load: model to load, init random if None
        :param save_model: name of the model in savefile and on tensorboard
        :param batch_size: size of batches
        :param learningrate: learning-rate
        """

        # init params
        self.save_model = save_model
        self.learningrate: float = learningrate
        self.batch_size: int = batch_size
        self.step: int = 0
        self.epoch: int = 0
        self.cur_best = 1000
        self.best_step = 0

        # create model
        self.model = DhSegment(
            [3, 4, 6, 4],
            in_channels=IN_CHANNELS,
            out_channel=OUT_CHANNELS,
            load_resnet_weights=True,
        )
        self.model = self.model.float()
        self.model.freeze_encoder()

        # load model if argument is None, it does nothing
        self.model.load(load)

        # set mean and std in a model for normalization
        self.model.means = torch.tensor((0.485, 0.456, 0.406))
        self.model.stds = torch.tensor((0.229, 0.224, 0.225))

        # set optimizer and loss_fn
        self.optimizer = Adam(
            self.model.parameters(), lr=learningrate, weight_decay=WEIGHT_DECAY
        )  # weight_decay=1e-4

        # load data
        dataset = NewsDataset()

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
            num_workers=DATALOADER_WORKER,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=DATALOADER_WORKER,
            drop_last=True,
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=DATALOADER_WORKER,
            drop_last=True,
        )

        # check for cuda
        self.device = args.cuda_device if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

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
        self.step = 0

        for self.epoch in range(1, epochs + 1):
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
                    with summary_writer.as_default():
                        tf.summary.scalar("train loss", loss.item(), step=self.step)
                        # tf.summary.scalar('batch mean', images.detach().cpu().mean(), step=self.step)
                        # tf.summary.scalar('batch std', images.detach().cpu().std(), step=self.step)
                        # tf.summary.scalar('target batch mean', targets.detach().cpu().float().mean(), step=self.step)
                        # tf.summary.scalar('target batch std', targets.detach().cpu().float().std(), step=self.step)

                    # update description
                    pbar.update(1)
                    pbar.set_postfix(**{"loss (batch)": loss.item()})

                    # delete data from gpu cache
                    del images, targets, loss, preds
                    torch.cuda.empty_cache()

                    if self.step % VAL_EVERY == 0:
                        loss, acc = self.validation()

                        # early stopping
                        if loss + (1 - acc) < self.cur_best:
                            # update cur_best value
                            self.cur_best = loss + (1 - acc)
                            self.best_step = self.step
                            print(
                                f"saved model because of early stopping with value {loss + (1 - acc)}"
                            )

                            # save the model
                            if self.save_model is not None:
                                self.model.save(self.save_model + "_best")

                    # log the step of current best model
                    # pylint: disable-next=not-context-manager
                    with summary_writer.as_default():
                        tf.summary.scalar(
                            "current best", self.best_step, step=self.step
                        )

            # save model at end of epoch
            self.model.save(self.save_model)

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

    def val_logging(self, loss, jaccard, accuracy, class_accs, test_validation: bool) -> None:
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
        with summary_writer.as_default():
            tf.summary.scalar("epoch", self.epoch, step=self.step)

            tf.summary.scalar(f"{environment}/loss", loss, step=self.step)
            tf.summary.scalar(f"{environment}/accuracy", accuracy, step=self.step)

            tf.summary.scalar(f"{environment}/jaccard score", jaccard, step=self.step)

            for i, acc in enumerate(class_accs):
                if not np.isnan(acc):
                    tf.summary.scalar(
                        f"multi-acc-{environment}/class {i}", acc, step=self.step
                    )

            tf.summary.image(
                f"image/{environment}-input",
                torch.permute(image.float().cpu(), (0, 2, 3, 1)),
                step=self.step,
            )
            tf.summary.image(
                f"image/{environment}-target",
                target.float().cpu()[None, :, :, None] / OUT_CHANNELS,
                step=self.step,
            )
            tf.summary.image(
                f"image/{environment}-prediction",
                pred.float().cpu()[:, :, :, None] / OUT_CHANNELS,
                step=self.step,
            )

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
        "--predict_image",
        "-i",
        type=str,
        default=PREDICT_IMAGE,
        help="path for full image prediction",
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
        "--predict-scale",
        "-p",
        type=float,
        default=PREDICT_SCALE,
        help="Downscaling factor of the predict image",
    )
    parser.add_argument(
        "--cuda-device", "-c", type=str, default="cuda:1", help="Cuda device string"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    PREDICT_SCALE = args.predict_scale
    PREDICT_IMAGE = args.predict_image

    # setup tensor board
    train_log_dir = "logs/runs/" + args.name
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    load_model = f"Models/model_{args.load}.pt" if args.load else None

    trainer = Trainer(
        load=load_model,
        save_model=f"Models/model_{args.name}",
        batch_size=args.batch_size,
        learningrate=args.lr,
    )

    trainer.train(epochs=args.epochs)
