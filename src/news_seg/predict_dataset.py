"""
module for Dataset class
"""
from __future__ import annotations

import os
from typing import Tuple

import torch
from PIL import Image
from PIL.Image import BICUBIC # pylint: disable=no-name-in-module
# pylint thinks torch has no name randperm this is wrong
# pylint: disable-next=no-name-in-module
from torch.utils.data import Dataset
from torchvision import transforms

from src.news_seg.utils import pad_image, calculate_padding

IMAGE_PATH = "data/images"
TARGET_PATH = "data/targets/"


class PredictDataset(Dataset):
    """
    A dataset class for the newspaper datasets
    """

    def __init__(
        self,
        image_path: str,
        scale: float,
        pad: Tuple[int, int]
    ) -> None:
        """
        load images and targets from folder
        :param preprocessing:
        :param data: uses this list instead of loading data from disc
        :param sort: sort file_names for testing purposes
        :param image_path: image path
        :param target_path: target path
        :param limit: limits the quantity of loaded images
        :param dataset: which dataset to expect. Options are 'transcribus' and 'HLNA2013' (europeaner newspaper project)
        """

        self.image_path = image_path
        self.scale = scale
        self.pad = pad

        self.file_names = []
        for file_name in os.listdir(image_path):
            if os.path.splitext(file_name)[1] != ".png" and os.path.splitext(file_name)[1] != ".jpg":
                continue
            self.file_names.append(file_name)

    def load_image(self, file: str) -> torch.Tensor:
        """
        Loads image and applies necessary transformation for prdiction.
        :param args: arguments
        :param file: path to image
        :return: Tensor of dimensions (BxCxHxW). In this case, the number of batches will always be 1.
        """
        image = Image.open(self.image_path + file).convert("RGB")
        shape = int(image.size[0] * self.scale), int(image.size[1] * self.scale)
        image = image.resize(shape, resample=BICUBIC)
        transform = transforms.PILToTensor()
        data: torch.Tensor = transform(image).float() / 255
        return data

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        """
        returns one datapoint
        :param item: number of the datapoint
        :return (tuple): torch tensor of image, torch tensor of annotation, tuple of mask
        """
        file_name = self.file_names[item]
        image = self.load_image(file_name)
        pad = calculate_padding(self.pad, image.shape, self.scale)
        image = pad_image(pad, image)
        return image, file_name

    def __len__(self) -> int:
        """
        standard len function
        :return: number of items in dateset
        """
        return len(self.file_names)
