"""Module for shared Dataset classes."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from src.cgprocess.shared.utils import (
    get_file_stems,
    initialize_random_split,
    prepare_file_loading,
)


class PageDataset(Dataset):
    """
    Dataset to handle page based data split.
    """

    def __init__(
        self,
        image_path: Path = Path("images"),
        dataset: str = "transkribus",
        file_stems: Union[List[str], None] = None,
    ) -> None:

        self.image_path = image_path
        self.dataset = dataset
        if file_stems:
            self.file_stems = file_stems
        else:
            extension, _ = prepare_file_loading(dataset)
            self.file_stems = get_file_stems(extension, image_path)

    def __len__(self) -> int:
        """
        Returns:
            int: number of items in dateset
        """
        return len(self.file_stems)

    def __getitem__(self, item: int) -> str:
        """
        returns one file stem
        """

        return self.file_stems[item]

    def random_split(
        self, ratio: Tuple[float, float, float]
    ) -> Tuple[PageDataset, PageDataset, PageDataset]:
        """
        splits the dataset in parts of size given in ratio

        Args:
            ratio(list): Ratio for train, val and test dataset

        Returns:
            tuple: Train, val and test PageDatasets
        """
        indices, splits = initialize_random_split(len(self), ratio)

        train_dataset = PageDataset(
            image_path=self.image_path,
            dataset=self.dataset,
            file_stems=np.array(self.file_stems)[indices[: splits[0]]].tolist(),
        )
        valid_dataset = PageDataset(
            image_path=self.image_path,
            dataset=self.dataset,
            file_stems=np.array(self.file_stems)[
                indices[splits[0] : splits[1]]
            ].tolist(),
        )
        test_dataset = PageDataset(
            image_path=self.image_path,
            dataset=self.dataset,
            file_stems=np.array(self.file_stems)[indices[splits[1] :]].tolist(),
        )

        return train_dataset, valid_dataset, test_dataset


class TrainDataset(Dataset, ABC):
    """
    Abstract Dataset Class for training. This Class can be supplied with a list of file stems.
    Otherwise, files to load will be determined based on the data path.
    Dataset Splitting on page level needs to be done beforehand.
    """

    def __init__(
        self,
        data_path: Path,
        limit: Optional[int] = None,
        data_source: str = "transkribus",
        file_stems: Optional[List[str]] = None,
        sort: bool = False,
        name: str = "default",
    ) -> None:
        """
        Args:
            data_path(Path): uses this list instead of loading data from disc
            limit(int): limits the quantity of loaded pages
            data_source(str): Name of image source, this influences the loading process.
            file_stems(list): File stems for images and targets
            sort(bool): sort file_names for testing purposes
            name(str): name of the dataset. E.g. train, val, test
        """
        self.image_path = data_path / "images"
        self.target_path = data_path / "targets"
        self.annotations_path = data_path / "annotations"
        self.data_source = data_source
        self.limit = limit
        self.name = name

        self.image_extension, self.get_file_name = prepare_file_loading(
            data_source
        )  # get_file_name is only for
        # compatability with europeana newspaper data
        if file_stems:
            self.file_stems = file_stems
        else:
            self.file_stems = get_file_stems(self.image_extension, self.image_path)
        if sort:
            self.file_stems.sort()

        if limit is not None:
            assert limit <= len(self.file_stems), (
                f"Provided limit with size {limit} is greater than dataset size"
                f"with size {len(self.file_stems)}."
            )
            self.file_stems = self.file_stems[:limit]

    def prepare_data(self) -> None:
        """Call extract data method if no preprocessed files are found, load data otherwise."""
        if not os.path.exists(self.target_path):
            print(f"creating {self.target_path}.")
            os.makedirs(self.target_path)  # type: ignore

        if self.files_missing():
            print("\n Initiating data extraction. \n")
            self.extract_data()
        else:
            print("\n Skipping data extraction. \n")

        self.get_data()

    def files_missing(self) -> bool:
        """
        Looks for json files with extracted data. If all of them are already present, data extraction should be skipped.
        Returns:
            bool: True if files are missing, False otherwise
        """
        file_list = np.array(os.listdir(self.target_path))
        for file_stem in self.file_stems:
            if f"{file_stem}.json" not in file_list:
                return True
        return False

    @abstractmethod
    def get_data(self) -> None:
        """Loads data and appends it to class attributes that store data until needed."""

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns:
            int: number of items in dateset
        """

    @abstractmethod
    def __getitem__(self, item: int) -> Any:
        """
        Returns:
            tuple: Input and target tensors
        """

    @abstractmethod
    def extract_data(self) -> None:
        """Extract data and save files inside the target directory."""
