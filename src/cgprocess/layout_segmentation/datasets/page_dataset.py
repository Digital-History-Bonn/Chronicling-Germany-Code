"""Module for PageDataset class. Handles page level Dataset Split."""
from __future__ import annotations

from typing import Tuple, Union, List

import numpy as np
import torch
# pylint thinks torch has no name randperm this is wrong
# pylint: disable-next=no-name-in-module
from torch import randperm
from torch.utils.data import Dataset

from src.cgprocess.layout_segmentation.datasets.train_dataset import IMAGE_PATH
from src.cgprocess.layout_segmentation.utils import prepare_file_loading, get_file_stems
from src.cgprocess.shared.utils import initialize_random_split


class PageDataset(Dataset):
    """
    Dataset to handle page based data split.
    """

    def __init__(
            self,
            image_path: str = IMAGE_PATH,
            dataset: str = "transkribus",
            file_stems: Union[List[str], None] = None
    ) -> None:

        if file_stems:
            self.file_stems = file_stems
        else:
            extension, _ = prepare_file_loading(dataset)
            self.file_stems = get_file_stems(extension, image_path)

    def __len__(self) -> int:
        """
        standard len function
        :return: number of items in dateset
        """
        return len(self.file_stems)

    def __getitem__(self, item: int) -> str:
        """
        returns one file stem
        :param item: number of the datapoint
        :return (tuple): torch tensor of image, torch tensor of annotation, tuple of mask
        """

        return self.file_stems[item]

    def random_split(
            self, ratio: Tuple[float, float, float]
    ) -> Tuple[PageDataset, PageDataset, PageDataset]:
        """
        splits the dataset in parts of size given in ratio
        :param ratio: list[float]:
        :return (tuple): tuple of PageDatasets
        """
        indices, splits = initialize_random_split(len(self), ratio)

        train_dataset = PageDataset(
            image_path="",
            dataset="",
            file_stems=np.array(self.file_stems)[indices[: splits[0]]].tolist()
        )
        valid_dataset = PageDataset(
            image_path="",
            dataset="",
            file_stems=np.array(self.file_stems)[indices[splits[0]: splits[1]]].tolist(),
        )
        test_dataset = PageDataset(
            image_path="",
            dataset="",
            file_stems=np.array(self.file_stems)[indices[splits[1]:]].tolist(),
        )

        return train_dataset, valid_dataset, test_dataset
