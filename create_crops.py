import os

import numpy as np
import torch
from PIL import Image  # type: ignore
from tqdm import tqdm

from preprocessing import Preprocessing

INPUT = "../prima/inputs/"
TARGETS = "../prima/targets/"

FOLDER = "crops/"


def main():
    preprocessing = Preprocessing()

    # read all file names
    paths = [f[:-4] for f in os.listdir(INPUT) if f.endswith(".tif")]
    print(f"{len(paths)=}")

    # iterate over files
    for file in tqdm(paths, desc='cropping images', unit='image'):
        # load image
        image = Image.open(f"{INPUT}{file}.tif").convert('RGB')

        # load target
        target = np.load(f"{TARGETS}pc-{file}.npy")

        assert image.size[1] == target.shape[0] and image.size[0] == target.shape[1], \
            f"image {file=} has shape {image.size}, but target has shape {target.shape}"

        # preprocess / create crops
        img_crops, tar_crops = preprocessing.preprocess(image, target)

        for i, (img_crop, tar_crop) in enumerate(zip(img_crops, tar_crops)):
            img_crop = torch.tensor(img_crop, dtype=torch.uint8)
            tar_crop = torch.tensor(tar_crop, dtype=torch.uint8)

            data = torch.cat((img_crop, tar_crop[None, :]), dim=0)

            torch.save(data, f"{FOLDER}{file}_{i}.pt")


if __name__ == '__main__':
    main()

