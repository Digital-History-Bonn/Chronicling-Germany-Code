"""module for validating dataset. Can be used to testwise load the entire Dataset"""
import argparse
import os

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.news_seg.news_dataset import NewsDataset
from src.news_seg.preprocessing import (
    Preprocessing,
)


def validate(args):
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


def count_classes(args: argparse.Namespace):
    """Load data with increasing amount of workers"""
    preprocessing = Preprocessing(scale=0.5, crop_factor=1, crop_size=512)
    dataset = NewsDataset(preprocessing, image_path=f"{args.data_path}images/",
                          target_path=f"{args.data_path}targets/",
                          dataset=args.dataset)
    dataset.augmentations = False

    loader = DataLoader(dataset, shuffle=False, num_workers=args.num_workers, batch_size=args.batch_size)
    class_counts = torch.empty((10), dtype=torch.long)
    for _, targets in tqdm(loader, desc="counting classes", unit="batches"):
        class_counts += torch.bincount(targets.flatten(), minlength=10)
    print(class_counts)
    print(class_counts / torch.sum(class_counts))


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
    parser.add_argument(
        "--classes",
        "-c",
        action="store_true",
        help="If activated, pixels of each class are counted.",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=1,
        help="Number of workers for the Dataloader",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=1,
        help="Batch size",
    )

    return parser.parse_args()


if __name__ == "__main__":
    parameter_args = get_args()
    if parameter_args.classes:
        count_classes(parameter_args)
    else:
        validate(parameter_args)
