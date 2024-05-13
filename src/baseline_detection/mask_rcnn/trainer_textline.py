"""Module to train Mask R-CNN Model"""

import os
from pathlib import Path
import argparse
from typing import Optional, Union, Dict

import numpy as np
import torch
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torchvision.models.detection import (
    FasterRCNN,
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN
)
from tqdm import tqdm

from src.baseline_detection.mask_rcnn.dataset_textline import CustomDataset
from src.baseline_detection.mask_rcnn.postprocessing import postprocess
from src.baseline_detection.mask_rcnn.utils import draw_prediction

LR = 0.00001


class Trainer:
    """Class to train models."""

    def __init__(
            self,
            model: FasterRCNN,
            traindataset: CustomDataset,
            testdataset: CustomDataset,
            optimizer: Optimizer,
            name: str,
            cuda: int = 0,
    ) -> None:
        """
        Trainer class to train models.

        Args:
            model: model to train
            traindataset: dataset to train on
            testdataset: dataset to validate model while trainings process
            optimizer: optimizer to use
            name: name of the model in save-files and tensorboard
            mask_prediction: Set True if you want to get masks predicted
            cuda: number of used cuda device
        """
        self.device = (
            torch.device(f"cuda:{cuda}")
            if torch.cuda.is_available() and cuda >= 0
            else torch.device("cpu")
        )
        print(f"using {self.device}")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.trainloader = DataLoader(
            traindataset, batch_size=1, shuffle=True, num_workers=16
        )
        self.testloader = DataLoader(
            testdataset, batch_size=1, shuffle=False, num_workers=16
        )

        self.bestavrgloss: Union[float, None] = None
        self.epoch = 0
        self.name = name

        # setup tensor board
        train_log_dir = f"{Path(__file__).parent.absolute()}/../../logs/runs/{self.name}"
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)  # type: ignore

        self.example_image, self.example_target = testdataset[0]
        self.train_example_image, self.train_example_target = traindataset[3]

    def save(self, name: str = "") -> None:
        """
        Save the model in models folder.

        Args:
            name: name of the model
        """
        os.makedirs(f"{Path(__file__).parent.absolute()}/../../models/", exist_ok=True)
        torch.save(
            self.model.state_dict(),
            f"{Path(__file__).parent.absolute()}/../../models/{name}",
        )

    def load(self, name: str = "") -> None:
        """
        Load the given model.

        Args:
            name: name of the model
        """
        self.model.load_state_dict(
            torch.load(f"{Path(__file__).parent.absolute()}/../../models/{name}.pt")
        )

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
        loss_classifier_lst = []
        loss_box_reg_lst = []
        loss_objectness_lst = []
        loss_rpn_box_reg_lst = []
        loss_masks_lst = []

        for img, target in tqdm(self.trainloader, desc="training"):
            img = img.to(self.device)
            target["boxes"] = target["boxes"][0].to(self.device)
            target["labels"] = target["labels"][0].to(self.device)
            target["masks"] = target["masks"][0].to(self.device)

            self.optimizer.zero_grad()
            output = model([img[0]], [target])
            loss = sum(v for v in output.values())
            loss.backward()
            self.optimizer.step()

            loss_lst.append(loss.detach().cpu().item())
            loss_classifier_lst.append(output["loss_classifier"].detach().cpu().item())
            loss_box_reg_lst.append(output["loss_box_reg"].detach().cpu().item())
            loss_objectness_lst.append(output["loss_objectness"].detach().cpu().item())
            loss_rpn_box_reg_lst.append(output["loss_rpn_box_reg"].detach().cpu().item())
            loss_masks_lst.append(output["loss_mask"].detach().cpu().item())

            del img, target, output, loss

        self.log_loss('Training',
                      loss=np.mean(loss_lst),
                      loss_classifier=np.mean(loss_classifier_lst),
                      loss_box_reg=np.mean(loss_box_reg_lst),
                      loss_objectness=np.mean(loss_objectness_lst),
                      loss_rpn_box_reg=np.mean(loss_rpn_box_reg_lst),
                      loss_masks=np.mean(loss_masks_lst)
                      )

        del (
            loss_lst,
            loss_classifier_lst,
            loss_box_reg_lst,
            loss_objectness_lst,
            loss_rpn_box_reg_lst,
            loss_masks_lst
        )

    def valid(self) -> float:
        """
        Validates current model on validation set.

        Returns:
            current loss
        """
        loss = []
        loss_classifier_lst = []
        loss_box_reg_lst = []
        loss_objectness_lst = []
        loss_rpn_box_reg_lst = []
        loss_masks_lst = []

        for img, target in tqdm(self.testloader, desc="validation"):
            img = img.to(self.device)
            target["boxes"] = target["boxes"][0].to(self.device)
            target["labels"] = target["labels"][0].to(self.device)
            target["masks"] = target["masks"][0].to(self.device)

            output = self.model([img[0]], [target])

            loss.append(sum(v for v in output.values()).cpu().detach())
            loss_classifier_lst.append(output["loss_classifier"].detach().cpu().item())
            loss_box_reg_lst.append(output["loss_box_reg"].detach().cpu().item())
            loss_objectness_lst.append(output["loss_objectness"].detach().cpu().item())
            loss_rpn_box_reg_lst.append(output["loss_rpn_box_reg"].detach().cpu().item())
            loss_masks_lst.append(output["loss_mask"].detach().cpu().item())

            del img, target, output

        self.log_loss('Valid',
                      loss=np.mean(loss),
                      loss_classifier=np.mean(loss_classifier_lst),
                      loss_box_reg=np.mean(loss_box_reg_lst),
                      loss_objectness=np.mean(loss_objectness_lst),
                      loss_rpn_box_reg=np.mean(loss_rpn_box_reg_lst),
                      loss_masks=np.mean(loss_masks_lst)
                      )

        self.log_examples('Training')
        self.log_examples('Valid')

        return np.mean(loss)

    def log_examples(self, dataset: str):
        """Predicts and logs a example image form the training- and from the validation set."""
        self.model.eval()

        example = self.train_example_image if dataset == 'Training' else self.example_image

        # predict example form training set
        pred = self.model([example.to(self.device)])[0]

        # move predictions to cpu
        pred["boxes"] = pred["boxes"].detach().cpu()
        pred["labels"] = pred["labels"].detach().cpu()
        pred["scores"] = pred["scores"].detach().cpu()
        pred["masks"] = pred["masks"].detach().cpu()

        # postprecess image (non maxima supression)
        pred = postprocess(pred, method='iom', threshold=.6)

        # draw image and log it to tensorboard
        result = draw_prediction(example, pred)
        self.writer.add_image(
            f"{dataset}/example",
            result[:, ::2, ::2],
            global_step=self.epoch
        )  # type: ignore

        self.model.train()

    def log_loss(self, dataset: str,
                 loss: float,
                 loss_classifier: float,
                 loss_box_reg: float,
                 loss_objectness: float,
                 loss_rpn_box_reg: float,
                 loss_masks: float
                 ):
        """
        Logs the loss values to tensorboard.

        Args:
            dataset: Name of the dataset the loss comes from ('Training' or 'Valid')
            loss: average over all loss
            loss_classifier: average classifier loss
            loss_box_reg: average box regression loss loss
            loss_objectness: average objectiveness loss
            loss_rpn_box_reg: average rpn box regression loss
            loss_masks: average masking loss
        """
        # logging
        self.writer.add_scalar(
            f"{dataset}/loss",
            loss,
            global_step=self.epoch
        )  # type: ignore

        self.writer.add_scalar(
            f"{dataset}/loss_classifier",
            loss_classifier,
            global_step=self.epoch
        )  # type: ignore

        self.writer.add_scalar(
            f"{dataset}/loss_box_reg",
            loss_box_reg,
            global_step=self.epoch
        )  # type: ignore

        self.writer.add_scalar(
            f"{dataset}/loss_objectness",
            loss_objectness,
            global_step=self.epoch
        )  # type: ignore

        self.writer.add_scalar(
            f"{dataset}/loss_rpn_box_reg",
            loss_rpn_box_reg,
            global_step=self.epoch
        )  # type: ignore

        self.writer.add_scalar(
            f"{dataset}/loss_masks",
            loss_masks,
            global_step=self.epoch
        )  # type: ignore

        self.writer.flush()  # type: ignore


def get_model(load_weights: Optional[str] = None) -> MaskRCNN:
    """
    Creates a FasterRCNN model for training, using the specified objective parameter.

    Args:
        load_weights: name of the model to load

    Returns:
        FasterRCNN model
    """
    params = {"box_detections_per_img": None}

    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                                  **params)

    if load_weights:
        model.load_state_dict(
            torch.load(
                f"{Path(__file__).parent.absolute()}/../../../models/" f"{load_weights}.pt"
            )
        )

    return model


def get_args() -> argparse.Namespace:
    """Defines arguments."""
    parser = argparse.ArgumentParser(description="training")

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
        "--load",
        "-l",
        type=str,
        default=None,
        help="name of a model to load",
    )

    parser.add_argument(
        "--cuda",
        "-c",
        type=int,
        default=-1,
        help="number of the cuda device (use -1 for cpu)",
    )

    parser.add_argument('--augmentations', "-a", action=argparse.BooleanOptionalAction)
    parser.set_defaults(augmentations=False)

    return parser.parse_args()


if __name__ == "__main__":
    from torchvision import transforms

    args = get_args()

    # check args
    if args.name == 'model':
        raise ValueError("Please enter a valid model name!")

    if args.epochs <= 0:
        raise ValueError("Please enter a valid number of epochs. Must be >= 0!")

    print(f'start training:\n'
          f'\tname: {args.name}\n'
          f'\tepochs: {args.epochs}\n'
          f'\tload: {args.load}\n'
          f'\tcuda: {args.cuda}\n')

    name = f"{args.name}{'_aug' if args.augmentations else ''}_e{args.epochs}"
    model = get_model(load_weights=args.load)

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
                p=0.1,
            ),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
            transforms.RandomGrayscale(p=0.1))

    traindataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../../../data/train_mask",
        transformation=transform,
    )

    validdataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../../../data/valid_mask"
    )

    print(f"{len(traindataset)=}")
    print(f"{len(validdataset)=}")

    optimizer = AdamW(model.parameters(), lr=LR)

    trainer = Trainer(model,
                      traindataset,
                      validdataset,
                      optimizer,
                      name,
                      cuda=args.cuda)
    trainer.train(args.epochs)
