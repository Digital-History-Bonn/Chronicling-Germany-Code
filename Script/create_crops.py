"""
Module to crop data. Crops will be saved in crops folder
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image  # type: ignore
from tqdm import tqdm  # type: ignore

from preprocessing import Preprocessing

INPUT = "DataBonn/Images/"
TARGETS = "DataBonn/targets/"

FOLDER = "cropsBonn/"


def main():
    """Load image and target data and saves single crops as torch tensor. Tensor contains 4 dimension, 3 for RGB image
    and target"""
    preprocessing = Preprocessing()

    if args.dataset == 'transcribus':
        extension = ".jpg"
        get_file_name = lambda name: f"{name}.npy"
    else:
        extension = ".tif"
        get_file_name = lambda name: f"pc-{file}.npy"

    # read all file names
    paths = [f[:-4] for f in os.listdir(INPUT) if f.endswith(extension)]
    print(f"{len(paths)=}")

    # iterate over files
    for file in tqdm(paths, desc='cropping images', unit='image'):
        # load image
        image = Image.open(f"{INPUT}{file}{extension}").convert('RGB')

        # load target
        target = np.load(f"{TARGETS}{get_file_name(file)}")

        assert image.size[1] == target.shape[0] and image.size[0] == target.shape[1], \
            f"image {file=} has shape {image.size}, but target has shape {target.shape}"

        # preprocess / create crops
        img_crops, tar_crops = preprocessing.preprocess(image, target)

        for i, (img_crop, tar_crop) in enumerate(zip(img_crops, tar_crops)):
            img_crop = torch.tensor(img_crop, dtype=torch.uint8)
            tar_crop = torch.tensor(tar_crop, dtype=torch.uint8)

            data = torch.cat((img_crop, tar_crop[None, :]), dim=0)

            torch.save(data, f"{FOLDER}{file}_{i}.pt")


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description='creates targets from annotation xmls')
    parser.add_argument('--dataset', '-d', type=str, default='transcribus', help='select dataset to crop '
                                                                                 '(transcribus, HLNA2013)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main()