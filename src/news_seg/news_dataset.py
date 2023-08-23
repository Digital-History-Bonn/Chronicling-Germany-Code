"""
module for Dataset class
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

# pylint thinks torch has no name randperm this is wrong
# pylint: disable-next=no-name-in-module
from torch import randperm
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from src.news_seg.preprocessing import Preprocessing

IMAGE_PATH = "data/images"
TARGET_PATH = "data/targets/"


class NewsDataset(Dataset):
    """
    A dataset class for the newspaper datasets
    """
    def __init__(self, image_path: str = IMAGE_PATH, target_path: str = TARGET_PATH,
                 data: Union[List[torch.Tensor], None] = None, limit: Union[int, None] = None,
                 dataset: str = "transcribus"):
        """
        load images and targets from folder
        :param image_path: image path
        :param target_path: target path
        :param limit: limits the quantity of loaded images
        :param dataset: which dataset to expect. Options are 'transcribus' and 'HLNA2013' (europeaner newspaper project)
        """

        self.preprocessing = Preprocessing()
        self.dataset = dataset
        self.image_path = image_path
        self.target_path = target_path

        if data:
            self.data = data
        else:
            # load data
            if self.dataset == "transcribus":
                extension = ".jpg"

                def get_file_name(name: str):
                    return f"{name}.npy"
            else:
                extension = ".tif"

                def get_file_name(name: str):
                    return f"pc-{name}.npy"

            # read all file names
            self.file_names = [f[:-4] for f in os.listdir(image_path) if f.endswith(extension)]

            if limit is not None:
                self.file_names = self.file_names[:limit]

            self.data: List[torch.Tensor] = []

            # iterate over files
            for file in tqdm(self.file_names, desc="cropping images", unit="image"):
                image, target = self.preprocessing.load(
                    f"{image_path}{file}{extension}", f"{target_path}{get_file_name(file)}", file
                )
                # preprocess / create crops
                crops = list(torch.tensor(self.preprocessing(image, target)))
                self.data += crops

        self.augmentations = True


    def __len__(self) -> int:
        """
        standard len function
        :return: number of items in dateset
        """
        return len(self.data)


    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns one datapoint
        :param item: number of the datapoint
        :return (tuple): torch tensor of image, torch tensor of annotation, tuple of mask
        """

        data: torch.Tensor = self.data[item]

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
        np_data = np.array(self.data)

        train_dataset = NewsDataset(
            image_path=self.image_path, target_path=self.target_path, data=list(np_data[indices[: splits[0]]])
        )
        test_dataset = NewsDataset(
            image_path=self.image_path, target_path=self.target_path, data=list(np_data[indices[splits[0]: splits[1]]])
        )
        valid_dataset = NewsDataset(
            image_path=self.image_path, target_path=self.target_path, data=list(np_data[indices[splits[1]:]])
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
