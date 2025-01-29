"""shared utility functions"""
import os
from pathlib import Path
from typing import Any, Tuple, List, Callable

import torch
from torch import randperm


def initialize_random_split(size: int, ratio: Tuple[float, float, float]) -> Tuple[Any, Tuple[int, int]]:
    """
    Args:
        size(int): Dataset size
        ratio(list): Ratio for train, val and test dataset

    Returns:
        tuple: List of randomly selected indices, as well as int tuple with two values that indicate the split points
        between train and val, as well as between val and test.
    """
    assert sum(ratio) == 1, "ratio does not sum up to 1."
    assert len(ratio) == 3, "ratio does not have length 3"
    assert (
            int(ratio[0] * size) > 0
            and int(ratio[1] * size) > 0
            and int(ratio[2] * size) > 0
    ), (
        "Dataset is to small for given split ratios for test and validation dataset. "
        "Test or validation dataset have size of zero."
    )
    splits = int(ratio[0] * size), int(ratio[0] * size) + int(
        ratio[1] * size
    )
    indices = randperm(
        size, generator=torch.Generator().manual_seed(42)
    ).tolist()
    return indices, splits


def get_file_stems(extension: str, image_path: Path) -> List[str]:
    """
    Returns file name without extension.
    Args:
        extension(str): extension of the files to be loaded
        image_path(Path): Ratio for train, val and test dataset

    Returns:
        list: file names.
    """
    file_names = [
        f[:-4] for f in os.listdir(image_path) if f.endswith(extension)
    ]
    assert len(file_names) > 0, (
        f"No Images in {image_path} with extension{extension} found. Make sure the "
        f"specified data source and path are correct."
    )
    return file_names


def prepare_file_loading(data_source: str) -> Tuple[str, Callable]:
    """Depending on the dataset this returns the correct extension string, as well as a function to get the
    file names for loading."""
    if data_source == "transkribus":
        # pylint: disable=duplicate-code
        extension = ".jpg"

        def get_file_name(name: str) -> str:
            return f"{name}.npz"

    elif data_source == "HLNA2013":
        extension = ".tif"

        def get_file_name(name: str) -> str:
            return f"pc-{name}.npz"

    else:
        extension = ".png"

        def get_file_name(name: str) -> str:
            return f"{name}.npz"
    return extension, get_file_name
