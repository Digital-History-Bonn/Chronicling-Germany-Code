"""shared utility functions"""

import json
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import torch

# pylint thinks torch has no name randperm this is wrong
# pylint: disable-next=no-name-in-module
from torch import randperm
from torch.utils.data import Dataset


def initialize_random_split(
    size: int, ratio: Tuple[float, float, float]
) -> Tuple[Any, Tuple[int, int]]:
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
    splits = int(ratio[0] * size), int(ratio[0] * size) + int(ratio[1] * size)
    indices = randperm(size, generator=torch.Generator().manual_seed(42)).tolist()
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
    file_names = [f[:-4] for f in os.listdir(image_path) if f.endswith(extension)]
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


def get_file_stem_split(
    custom_split_file: Optional[str],
    split_ratio: Tuple[float, float, float],
    page_dataset: Dataset,
) -> tuple[List[str], List[str], List[str]]:
    """
    Creates dataset split or initializes it from a config file
    """
    # todo: merge this with other methods
    if custom_split_file:
        with open(custom_split_file, "r", encoding="utf-8") as file:
            split = json.load(file)
            train_file_stems = split["Training"]
            val_file_stems = split["Validation"]
            test_file_stems = split["Test"]
            print(
                f"custom page level split with train size {len(train_file_stems)}, val size"
                f" {len(val_file_stems)} and test size {len(test_file_stems)}"
            )
    else:
        train_pages, validation_pages, test_pages = page_dataset.random_split(
            split_ratio
        )
        train_file_stems = train_pages.file_stems
        val_file_stems = validation_pages.file_stems
        test_file_stems = test_pages.file_stems

        with open("custom_split_file.json", "w", encoding="utf8") as file:
            json.dump(
                {
                    "Training": train_file_stems,
                    "Validation": val_file_stems,
                    "Test": test_file_stems,
                },
                file,
            )
    return test_file_stems, train_file_stems, val_file_stems


def xml_polygon_to_polygon_list(polygon_string: str) -> List[List[int]]:
    """
    Splits xml polygon coordinate string to create a polygon, this being a list of coordinate pairs.
    """
    return [list(map(int, point.split(","))) for point in polygon_string.split()]


def get_bbox(
    points: Union[torch.Tensor],  # type: ignore
) -> Tuple[int, int, int, int]:
    """
    Creates a bounding box around all given points.

    Args:
        points: p.ndarray of shape (N x 2) containing a list of points

    Returns:
        coordinates of bounding box in the format (x min, y_min, x_max, y_mx)
    """
    x_max, x_min = points[:, 0].max(), points[:, 0].min()
    y_max, y_min = points[:, 1].max(), points[:, 1].min()

    return x_min.item(), y_min.item(), x_max.item(), y_max.item()  # type: ignore


def enforce_image_limits(polygon: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Limit polygon points to coordinates inside given shape.
    """
    polygon[polygon < 0] = 0
    polygon[:, 0][polygon[:, 0] > shape[0]] = shape[0]
    polygon[:, 1][polygon[:, 1] > shape[1]] = shape[1]
    return polygon
