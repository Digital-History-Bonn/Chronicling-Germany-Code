"""Splits data into training, validation and test split."""

import glob
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


def copy(folder: str, split: List[str], name: str) -> None:
    """
    Copies the files form a given split to a new folder.

    Args:
        folder: folder to copy the files form
        split: list of files to copy
        name: name of the new folder (should be 'train', 'valid', 'test')
    """
    os.makedirs(f"{folder}/../{name}")

    for f in tqdm(split, desc="copying"):
        numb = f.split(os.sep)[-1]
        shutil.copytree(f, f"{folder}/../{name}/{numb}")


def split_dataset(folder: str, split: Tuple[float, float, float]) -> None:
    """
    Splits the dataset into training, validation and test split.

    Args:
        folder: folder with train dataset
        split: Tuple with partitions of splits (should sum up to 1)
    """
    # seed the process for reproducibility
    random.seed(42)

    # get folder with datapoints form train dataset
    files = list(glob.glob(f"{folder}/*"))
    print(f"{folder}/*")
    print(f"{len(files)=}")

    # shuffle
    shuffled_list = random.sample(files, len(files))

    # calc split indices
    n = len(files)
    s1, s2 = int(n * split[0]), int(n * (split[0] + split[1]))

    # create splits
    train = shuffled_list[:s1]
    valid = shuffled_list[s1:s2]
    test = shuffled_list[s2:]

    print(f"{len(train)=}")
    print(f"{len(valid)=}")
    print(f"{len(test)=}")

    # copy data in 3 new folder
    copy(folder, train, "train_mask")
    copy(folder, valid, "valid_mask")
    copy(folder, test, "test_mask")


if __name__ == "__main__":
    split_dataset(f"{Path(__file__).parent.absolute()}/../../data/preprocessed",
                  (0.8, 0.1, 0.1))
