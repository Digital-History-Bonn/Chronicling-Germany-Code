"""Module to train baseline models."""

import argparse
import os
from typing import Dict, Tuple, Union

from PIL import Image
import numpy as np
import torch
from monai.losses import DiceLoss
from monai.networks.nets import BasicUNet
from torch import nn
from torch.nn import DataParallel, MSELoss
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torchvision import transforms
from tqdm import tqdm

from cgprocess.baseline_detection.dataset import CustomDataset as Dataset
from cgprocess.baseline_detection.utils import adjust_path, set_seed

Image.MAX_IMAGE_PIXELS = 500000000

LR = 0.0001


class MultiTargetLoss(nn.Module):
    """
    Custom loss for multi task semantic segmentation.

    Oriented on Transformer (https://arxiv.org/abs/2102.11838)
    """

    def __init__(self, scaling: float = 0.01):
        """
        Custom loss for multi-task semantic segmentation.

        Args:
            scaling: waiting between ascender- and descender loss and baseline- and limit loss
        """
        super().__init__()
        self.scaling = scaling
        self.dice = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        self.mse = MSELoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[float, float, float, float, float]:
        """
        Forward pass for loss.

        Args:
            pred: tensor with predictions (batch, channel, width, height)
            target: tensor with targets (batch, channel, width, height)

        Returns:
            overall loss, individual losses (ascender, descender, baseline, limits)
        """
        ascender_loss = self.mse(
            pred[:, 0, :, :].float() * target[:, 2, :, :].float(),
            target[:, 0, :, :].float(),
        )
        descender_loss = self.mse(
            pred[:, 1, :, :].float() * target[:, 2, :, :].float(),
            target[:, 1, :, :].float(),
        )
        baseline_loss = self.dice(pred[:, 2:4, :, :], target[:, 2, None, :, :])
        limits_loss = self.dice(pred[:, 4:6, :, :], target[:, 3, None, :, :])

        loss = (
            self.scaling * (ascender_loss + descender_loss)
            + baseline_loss
            + limits_loss
        )
        return loss, ascender_loss, descender_loss, baseline_loss, limits_loss


class Trainer:
    """Class to train models."""

    def __init__(
        self,
        model: nn.Module,
        traindataset: Dataset,
        testdataset: Dataset,
        optimizer: Optimizer,
        name: str,
        gpu_count: int = 0,
        worker_count: int = 1,
        batch_size: int = 32,
    ) -> None:
        """
        Trainer class to train models.

        Args:
            model: model to train
            traindataset: dataset to train on
            testdataset: dataset to validate model while trainings process
            optimizer: optimizer to use
            name: name of the model in save-files and tensorboard
            gpu_count: number of used cuda device
        """
        print(f"{torch.cuda.is_available()=}")
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and gpu_count >= 0
            else torch.device("cpu")
        )
        print(f"using {self.device}")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = MultiTargetLoss()
        self.softmax = nn.Softmax(dim=1)

        self.trainloader = DataLoader(
            traindataset,
            batch_size=batch_size * gpu_count,
            shuffle=True,
            num_workers=worker_count,
            persistent_workers=True,
            drop_last=True,
        )
        self.testloader = DataLoader(
            testdataset, batch_size=1 * gpu_count, shuffle=False, num_workers=16, persistent_workers=True
        )

        self.bestavrgloss: Union[float, None] = None
        self.epoch = 0
        self.name = name

        # setup tensorboard
        train_log_dir = f"logs/runs/{self.name}"
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)  # type: ignore

        self.example_image, self.example_target = testdataset[2]
        self.train_example_image, self.train_example_target = traindataset[0]

    def save(self, name: str = "") -> None:
        """
        Save the model in models folder.

        Args:
            name: name of the model
        """
        os.makedirs("models/", exist_ok=True)
        torch.save(
            self.model.module.state_dict(),
            f"models/{name}",
        )

    def load(self, name: str = "") -> None:
        """
        Load the given model.

        Args:
            name: name of the model
        """
        self.model.module.load_state_dict(torch.load(f"models/{name}.pt"))

    def train(self, epoch: int) -> None:
        """
        Train model for given number of epochs.

        Args:
            epoch: number of epochs
        """
        for self.epoch in range(1, epoch + 1):
            print(f"start epoch {self.epoch}:")
            self.train_epoch()
            avgloss = self.valid()

            # early stopping
            if self.bestavrgloss is None or self.bestavrgloss > avgloss:
                self.bestavrgloss = avgloss
                self.save(f"{self.name}_es.pt")

        # save model after training
        self.save(f"{self.name}_end.pt")

    def train_epoch(self) -> None:
        """Trains one epoch."""
        loss_lst = []
        baseline_loss_lst = []
        ascender_loss_lst = []
        descender_loss_lst = []
        limits_loss_lst = []

        for images, targets in tqdm(self.trainloader, desc="training"):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(images)
            loss, asc_loss, desc_loss, baseline_loss, limits_loss = self.loss_fn(
                output, targets
            )
            loss.backward()
            self.optimizer.step()

            loss_lst.append(loss.cpu().detach())
            baseline_loss_lst.append(baseline_loss.cpu().detach())
            ascender_loss_lst.append(asc_loss.cpu().detach())
            descender_loss_lst.append(desc_loss.cpu().detach())
            limits_loss_lst.append(limits_loss.cpu().detach())

            del (
                images,
                targets,
                output,
                loss,
                baseline_loss,
                asc_loss,
                desc_loss,
                limits_loss,
            )

        self.log_loss(
            "Training",
            loss=np.mean(loss_lst),
            baseline_loss=np.mean(baseline_loss_lst),
            ascender_loss=np.mean(ascender_loss_lst),
            descender_loss=np.mean(descender_loss_lst),
            limits_loss=np.mean(limits_loss_lst),
        )

        del loss_lst

    def valid(self) -> float:
        """
        Validates current model on validation set.

        Returns:
            current loss
        """
        loss_lst = []
        baseline_loss_lst = []
        ascender_loss_lst = []
        descender_loss_lst = []
        limits_loss_lst = []

        for images, targets in tqdm(self.testloader, desc="validation"):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(images)
            loss, asc_loss, desc_loss, baseline_loss, limits_loss = self.loss_fn(
                output, targets
            )

            loss_lst.append(loss.cpu().detach())
            baseline_loss_lst.append(baseline_loss.cpu().detach())
            ascender_loss_lst.append(asc_loss.cpu().detach())
            descender_loss_lst.append(desc_loss.cpu().detach())
            limits_loss_lst.append(limits_loss.cpu().detach())

            del (
                images,
                targets,
                output,
                loss,
                baseline_loss,
                asc_loss,
                desc_loss,
                limits_loss,
            )

        self.log_loss(
            "Valid",
            loss=np.mean(loss_lst),
            baseline_loss=np.mean(baseline_loss_lst),
            ascender_loss=np.mean(ascender_loss_lst),
            descender_loss=np.mean(descender_loss_lst),
            limits_loss=np.mean(limits_loss_lst),
        )

        self.log_examples("Training")
        self.log_examples("Valid")

        return np.mean(loss_lst)  # type: ignore

    def log_examples(self, dataset: str) -> None:
        """
        Predicts and logs a example image form the training- and from the validation set.

        Args:
            dataset: dataset to log
        """
        self.model.eval()

        example = (
            self.train_example_image if dataset == "Training" else self.example_image
        )

        # predict example form training set
        pred = self.model(example[None].to(self.device))

        ascenders = pred[:, 0, :, :].clip(min=0) / pred[:, 0, :, :].max()
        descenders = pred[:, 1, :, :].clip(min=0) / pred[:, 1, :, :].max()
        baselines = self.softmax(pred[:, 2:4, :, :])
        limits = self.softmax(pred[:, 4:, :, :])

        self.log_image(
            dataset=dataset,
            ascenders=ascenders,
            descenders=descenders,
            baselines=baselines[:, 1],
            limits=limits[:, 1],
        )

        self.model.train()

    def log_image(self, dataset: str, **kwargs: Dict[str, torch.Tensor]) -> None:
        """
        Logs given images under the given dataset label.

        Args:
            dataset: dataset to log the images under ('Training' or 'Validation')
            kwargs: Dict with names (keys) and images (images) to log
        """
        for key, image in kwargs.items():
            # log in tensorboard
            self.writer.add_image(
                f"{dataset}/{key}",
                image[:, ::2, ::2],  # type: ignore
                global_step=self.epoch,
            )  # type: ignore

        self.writer.flush()  # type: ignore

    def log_loss(self, dataset: str, **kwargs: Dict[str, float]) -> None:
        """
        Logs the loss values to tensorboard.

        Args:
            dataset: Name of the dataset the loss comes from ('Training' or 'Valid')
            kwargs: dict with loss names (keys) and loss values (values)

        """
        # logging
        for key, value in kwargs.items():
            self.writer.add_scalar(
                f"{dataset}/{key}", value, global_step=self.epoch
            )  # type: ignore

        self.writer.flush()  # type: ignore


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with call arguments
    """
    parser = argparse.ArgumentParser(description="training")
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="test",
        help="Name of the model, for saving and logging",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=2,
        help="Number of epochs",
    )

    parser.add_argument(
        "--train_data",
        "-t",
        type=str,
        default=None,
        help="path for folder with images jpg files and annotation xml files to train the model.",
    )

    parser.add_argument(
        "--valid_data",
        "-v",
        type=str,
        default=None,
        help="path for folder with images jpg files and annotation xml files to validate the model.",
    )
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        default=-1,
        help="number of the gpu devices to use. (use -1 for cpu)",
    )

    # pylint: disable=duplicate-code
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Seeding number for random generators.",
    )
    parser.add_argument(
        "--num-worker",
        "-w",
        type=int,
        default=8,
        help="Number of workers for train dataloader.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="Batchsize for training.",
    )

    parser.add_argument("--augmentations", "-a", action=argparse.BooleanOptionalAction)
    parser.set_defaults(augmentations=False)

    return parser.parse_args()


def main() -> None:
    """Trains a model with given parameters."""
    args = get_args()

    train_data = adjust_path(args.train_data)
    valid_data = adjust_path(args.valid_data)

    print(f"{args =}")

    # set random seed
    set_seed(seed=args.seed)

    # check args
    if train_data is None:
        raise ValueError("Please enter a valid path to training data!")

    if valid_data is None:
        raise ValueError("Please enter a valid path to validation data!")

    if args.name == "model":
        raise ValueError("Please enter a valid model name!")

    if args.epochs <= 0:
        raise ValueError("Please enter a valid number of epochs must be >= 0!")

    print(
        f"start training:\n"
        f"\tname: {args.name}\n"
        f"\tepochs: {args.epochs}\n"
        f"\tgpu count: {args.gpu}\n"
    )

    # init model
    name = (
        f"{args.name}_baseline" f"{'_aug' if args.augmentations else ''}_e{args.epochs}"
    )
    model = DataParallel(BasicUNet(spatial_dims=2, in_channels=3, out_channels=6))

    # init transformations for augmentation
    transform = None
    if args.augmentations:
        transform = torch.nn.Sequential(
            transforms.RandomApply(
                torch.nn.ModuleList(
                    [transforms.ColorJitter(brightness=(0.5, 1.5), saturation=(0, 2))]
                ),
                p=0.1,
            ),
            transforms.RandomApply(
                torch.nn.ModuleList(
                    [transforms.GaussianBlur(kernel_size=9, sigma=(2, 10))]
                ),
                p=0.3,
            ),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
            transforms.RandomGrayscale(p=1),
        )

    # init datasets
    traindataset = Dataset(
        train_data,
        augmentations=transform,
    )

    validdataset = Dataset(valid_data, cropping=False)

    print(f"{len(traindataset)=}")
    print(f"{len(validdataset)=}")

    # init optimizer and trainer
    optimizer = AdamW(model.parameters(), lr=LR)
    trainer = Trainer(
        model, traindataset, validdataset, optimizer, name, gpu_count=args.gpu, worker_count=args.num_worker, batch_size=args.batch_size
    )

    # start training
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
