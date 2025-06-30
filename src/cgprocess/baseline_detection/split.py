"""Splits preprocessed data into training, validation and test split."""

import argparse
import json
import os
import shutil
from typing import List

from tqdm import tqdm

from cgprocess.OCR.shared.utils import adjust_path


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    # pylint: disable=duplicate-code
    parser = argparse.ArgumentParser(description="preprocess")

    parser.add_argument(
        "--image_dir",
        "-i",
        type=str,
        default=None,
        help="path for folder with images. Images need to be jpg.",
    )

    parser.add_argument(
        "--annotation_dir",
        "-a",
        type=str,
        default=None,
        help="path for folder with layout xml files.",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="path to the folder where to save the preprocessed trainings targets",
    )

    return parser.parse_args()


def copy(folder: str, split: List[str]) -> None:
    """
    Copies the files form a given split to a new folder.

    Args:
        folder: folder to copy the files form
        split: list of files to copy
    """
    # pylint: disable=duplicate-code
    os.makedirs(folder, exist_ok=True)

    for f in tqdm(split, desc="copying"):
        shutil.copy(f, f"{folder}/{os.path.basename(f)}")


def main() -> None:
    """Splits the dataset into training, validation and test split."""
    # pylint: disable=duplicate-code
    args = get_args()
    output_dir = adjust_path(args.output_dir)
    image_dir = adjust_path(args.image_dir)
    annotation_dir = adjust_path(args.annotation_dir)

    with open("neurips-split.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    train_files = data.get("Training", [])
    valid_files = data.get("Validation", [])
    test_files = data.get("Test", [])

    print(f"{len(train_files)=}")
    print(f"{len(valid_files)=}")
    print(f"{len(test_files)=}")

    # copy data in 3 new folder
    copy(f"{output_dir}/train", [f"{image_dir}/{x}.jpg" for x in train_files])
    copy(f"{output_dir}/train", [f"{annotation_dir}/{x}.npz" for x in train_files])

    copy(f"{output_dir}/valid", [f"{image_dir}/{x}.jpg" for x in valid_files])
    copy(f"{output_dir}/valid", [f"{annotation_dir}/{x}.npz" for x in valid_files])

    copy(f"{output_dir}/test", [f"{image_dir}/{x}.jpg" for x in test_files])
    copy(f"{output_dir}/test", [f"{annotation_dir}/{x}.npz" for x in test_files])


if __name__ == "__main__":
    main()
