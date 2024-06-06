"""Module for PageDataset class. Handles page level Dataset Split."""
from __future__ import annotations

from typing import Tuple, Union, List

import numpy as np
import torch
# pylint thinks torch has no name randperm this is wrong
# pylint: disable-next=no-name-in-module
from torch import randperm
from torch.utils.data import Dataset

from src.layout_segmentation.datasets.train_dataset import IMAGE_PATH
from src.layout_segmentation.utils import prepare_file_loading, get_file_stems


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
        # pylint: disable=duplicate-code
        assert sum(ratio) == 1, "ratio does not sum up to 1."
        assert len(ratio) == 3, "ratio does not have length 3"
        assert (
                int(ratio[0] * len(self)) > 0
                and int(ratio[1] * len(self)) > 0
                and int(ratio[2] * len(self)) > 0
        ), (
            "Dataset is to small for given split ratios for test and validation dataset. "
            "Test or validation dataset have size of zero."
        )

        splits = int(ratio[0] * len(self)), int(ratio[0] * len(self)) + int(
            ratio[1] * len(self)
        )

        indices = randperm(
            len(self), generator=torch.Generator().manual_seed(42)
        ).tolist()

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
