"""module for validating dataset. Can be used to testwise load the entire Dataset"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cgprocess.layout_segmentation.datasets.train_dataset import TrainDataset
from src.cgprocess.layout_segmentation.processing.preprocessing import Preprocessing


def validate(args: argparse.Namespace) -> None:
    """Load data to validate shape"""
    # read all file names
    preprocessing = Preprocessing()
    dataset = args.dataset
    image_path = f"{args.data_path}images/"
    target_path = f"{args.data_path}targets/"

    # load data
    if dataset == "transcribus":
        extension = ".jpg"

        def get_file_name(name: str) -> str:
            return f"{name}.npz"

    else:
        extension = ".tif"

        def get_file_name(name: str) -> str:
            return f"pc-{name}.npz"

    file_names = [f[:-4] for f in os.listdir(image_path) if f.endswith(extension)]
    assert len(file_names) > 0, (
        f"No Images in {image_path} with extension{extension} found. Make sure the "
        f"specified dataset and path are correct."
    )
    file_names.sort()

    # iterate over files
    for file in tqdm(file_names, desc="cropping images", unit="image"):
        # pylint: disable-next=bare-except
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
                    f"image {file=} has shape w:{image.size[0]}, h: {image.size[1]}, "
                    f"but target has shape w:{target.shape[1]}, "
                    f"h: {target.shape[0]}"
                )
        # pylint: disable-next=bare-except
        except:
            print(f"{file}")


def count_classes(args: argparse.Namespace) -> None:
    """Load data with increasing amount of workers"""
    parameter_args = get_args()
    data_path = Path(parameter_args.data_path)
    class_counts = torch.zeros((10), dtype=torch.long)

    for path in tqdm(data_path.iterdir(), desc="counting classes", unit=" files"):
        if path.suffix == ".npz":
            img = np.load(path)["array"]
            class_counts += torch.bincount(torch.tensor(img.flatten()), minlength=10)
    print(class_counts)
    print(class_counts / torch.sum(class_counts))


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="validate dataset")
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        dest="data_path",
        default=None,
        help="path for folder with folders 'images' and 'targets'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parameter_args = get_args()
    count_classes(parameter_args)
