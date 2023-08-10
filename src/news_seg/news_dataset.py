"""
module for Dataset class
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch  # type: ignore

# pylint thinks torch has no name randperm this is wrong
# pylint: disable-next=no-name-in-module
from torch import randperm  # type: ignore
from torch.utils.data import Dataset  # type: ignore
from torchvision import transforms  # type: ignore

PATH = "crops/"


class NewsDataset(Dataset):
    """
    A dataset class for the newspaper datasets
    """

    def __init__(self, path: str = PATH, files: Union[List[str], None] = None, limit: Union[int, None] = None):
        """
        Dataset object
        load images and targets from folder

        :param path: path to folders with images and targets
        """
        # path to the over all folder
        self.path = path

        # list paths of images and targets
        if files is None:
            self.file_names: List[str] = [
                f[:-3] for f in os.listdir(f"{path}") if f.endswith(".pt")
            ]
        else:
            self.file_names = files

        if limit is not None:
            self.file_names = self.file_names[:limit]

        self.augmentations = True

    def __len__(self) -> int:
        """
        standard len function
        :return: number of items in dateset
        """
        return len(self.file_names)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns one datapoint
        :param item: number of the datapoint
        :return (tuple): torch tensor of image, torch tensor of annotation, tuple of mask
        """
        # load data
        data = torch.load(f"{self.path}{self.file_names[item]}.pt")

        # do augmentations
        if self.augmentations:
            augmentations = self.get_augmentations()
            data = augmentations["default"](data)
            img = augmentations["images"](data[:-1]).float() / 255
            # img = (img + (torch.randn(img.shape) * 0.05)).clip(0, 1)     # originally 0.1
        else:
            img = data[:-1].float() / 255

        return img, data[-1].long()

    def random_split(
            self, ratio: Tuple[float, float, float]
    ) -> Tuple[NewsDataset, NewsDataset, NewsDataset]:
        """
        splits the dataset in parts of size given in ratio
        :param ratio: list[float]:
        :return (list): list of NewsDatasets
        """
        assert sum(ratio) == 1, "ratio does not sum up to 1."
        assert len(ratio) == 3, "ratio does not have length 3"

        splits = int(ratio[0] * len(self)), int(ratio[0] * len(self)) + int(
            ratio[1] * len(self)
        )

        indices = randperm(
            len(self), generator=torch.Generator().manual_seed(42)
        ).tolist()
        nd_paths = np.array(self.file_names)

        train_dataset = NewsDataset(
            path=self.path, files=list(nd_paths[indices[: splits[0]]])
        )
        test_dataset = NewsDataset(
            path=self.path, files=list(nd_paths[indices[splits[0]: splits[1]]])
        )
        valid_dataset = NewsDataset(
            path=self.path, files=list(nd_paths[indices[splits[1]:]])
        )

        return train_dataset, test_dataset, valid_dataset

    @staticmethod
    def get_augmentations() -> Dict[str, transforms.Compose]:
        """Defines transformations"""
        return {
            "default": transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(180),
                ]
            ),
            "images": transforms.RandomApply(
                [transforms.Compose([transforms.GaussianBlur(5, (0.1, 1.5))])], p=0.5
            ),
        }  # originally 0.8
