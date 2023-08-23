"""
Module to crop data. Crops will be saved in crops folder
"""
import argparse
import os

import torch
from tqdm import tqdm

from src.news_seg.preprocessing import Preprocessing


def main():
    """Load image and target data and saves single crops as torch tensor. Tensor contains 4 dimension, 3 for RGB image
    and target"""
    preprocessing = Preprocessing()

    # pylint: disable=duplicate-code
    if args.dataset == "transcribus":
        extension = ".jpg"

        def get_file_name(name: str):
            return f"{name}.npy"
    else:
        extension = ".tif"

        def get_file_name(name: str):
            return f"pc-{name}.npy"

    # read all file names
    paths = [f[:-4] for f in os.listdir(args.images) if f.endswith(extension)]
    print(f"{len(paths)=}")

    # iterate over files
    for file in tqdm(paths, desc="cropping images", unit="image"):
        image, target = preprocessing.load(
            f"{args.images}{file}{extension}", f"{args.targets}{get_file_name(file)}", f"{file}"
        )
        # preprocess / create crops
        img_crops, tar_crops = preprocessing(image, target)

        for i, (img_crop, tar_crop) in enumerate(zip(img_crops, tar_crops)):
            img_crop = torch.tensor(img_crop, dtype=torch.uint8)
            tar_crop = torch.tensor(tar_crop, dtype=torch.uint8)

            data = torch.cat((img_crop, tar_crop[None, :]), dim=0)

            torch.save(data, f"{args.output}{file}_{i}.pt")


def get_args() -> argparse.Namespace:
    """defines arguments"""
    # pylint: disable=locally-disabled, duplicate-code
    parser = argparse.ArgumentParser(description="creates targets from annotation xmls")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="transcribus",
        help="select dataset to crop " "(transcribus, HLNA2013)",
    )
    parser.add_argument(
        "--images",
        "-i",
        type=str,
        default="data/images/",
        help="Image input folder",
    )
    parser.add_argument(
        "--targets",
        "-t",
        type=str,
        default="data/targets/",
        help="Target input folder",
    )
    parser.add_argument(
        "--output",
        "-c",
        type=str,
        default="output/",
        help="Output folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main()
