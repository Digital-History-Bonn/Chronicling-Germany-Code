import argparse
import os
from time import time  # type: ignore
from typing import List

import torch
from torch.utils.data import DataLoader  # type: ignore
from tqdm import tqdm

from src.news_seg.news_dataset import NewsDataset  # type: ignore
from src.news_seg.preprocessing import (
    CROP_FACTOR,
    CROP_SIZE,  # type: ignore
    Preprocessing,
)


def validate():
    """Load data to validate shape"""
    # read all file names
    preprocessing = Preprocessing()
    dataset = args.dataset
    image_path = f"{args.data_path}images/"
    target_path = f"{args.data_path}targets/"

    data: List[torch.Tensor] = []
    if data:
        data = data
    else:
        # load data
        if dataset == "transcribus":
            extension = ".jpg"

            def get_file_name(name: str) -> str:
                return f"{name}.npy"

        else:
            extension = ".tif"

            def get_file_name(name: str) -> str:
                return f"pc-{name}.npy"

    file_names = [f[:-4] for f in os.listdir(image_path) if f.endswith(extension)]
    assert len(file_names) > 0, (
        f"No Images in {image_path} with extension{extension} found. Make sure the "
        f"specified dataset and path are correct."
    )
    file_names.sort()

    # iterate over files
    for file in tqdm(file_names, desc="cropping images", unit="image"):
        try:
            image, target = preprocessing.load(
                f"{image_path}{file}{extension}",
                f"{target_path}{get_file_name(file)}",
                file,
                dataset,
            )
            if not (
                image.size[1] == target.shape[0] and image.size[0] == target.shape[1]
            ):
                print(
                    f"image {file=} has shape w:{image.size[0]}, h: {image.size[1]}, but target has shape w:{target.shape[1]}, "
                    f"h: {target.shape[0]}"
                )
        except:
            print(f"{file}")


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="train")
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        dest="data_path",
        default=None,
        help="path for folder with folders 'images' and 'targets'",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="transcribus",
        help="which dataset to expect. Options are 'transcribus' and 'HLNA2013' "
        "(europeaner newspaper project)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    validate()
