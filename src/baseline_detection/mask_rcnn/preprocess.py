"""Preprocess for the Newspaper dataset to predict Textlines."""

import glob
import os
from pathlib import Path
from typing import List, Dict, Union, Tuple

import torch
from shapely import LineString
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from skimage import draw
from bs4 import PageElement
from skimage import io
from tqdm import tqdm
import re

from src.baseline_detection.utils import extract


def create_baseline_target(shape: Tuple[int, int],
                           baselines: List[torch.Tensor],
                           mask_regions: List[torch.Tensor]) -> np.ndarray:
    """
    Creates a baseline target for given shape.

    Args:
        shape: shape of target
        baselines: list of baselines
        mask_regions: regions to mask in input image,
                      because we know there are no baseline from layout

    Returns:
        numpy array representing target for baseline prediction
    """
    # Create a blank target filled with ones (white)
    target = np.zeros((*shape, 2), dtype=np.uint8)
    target[:, :, 1] = 1  # mask with default value true

    # create PIL draw
    baseline_img = Image.fromarray(target[:, :, 0])
    baseline_draw = ImageDraw.Draw(baseline_img)

    # Draw baselines
    for baseline in baselines:
        line = LineString(torch.flip(baseline, dims=[1]))
        baseline_draw.line(line.coords, fill=1, width=1)

    target[:, :, 0] = np.array(baseline_img)

    for mask_region in mask_regions:
        # draw mask to remove not text regions
        if len(mask_region) >= 3:
            rr, cc = draw.polygon(mask_region[:, 0], mask_region[:, 1], shape=shape)
            target[rr, cc, 1] = 0

    return np.array(target)


def get_tag(textregion: PageElement) -> str:
    """
    Returns the tag of the given textregion.

    Args:
        textregion: PageElement of Textregion

    Returns:
        Given tag of that Textregion
    """
    desc = textregion['custom']
    match = re.search(r"\{type:.*;\}", desc)
    if match is None:
        return 'UnknownRegion'
    return match.group()[6:-2]


def rename_files(folder_path: str) -> None:
    """
    Renames all files and folders in given folder by replacing 'ö' with 'oe'.

    Args:
        folder_path: path to folder
    """
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file
    for filename in files:
        # Check if the file name contains 'ö'
        if 'ö' in filename:
            # Replace 'ö' with 'oe' in the filename
            new_filename = filename.replace('ö', 'oe')

            # Construct the full old and new paths
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed {filename} to {new_filename}")


def main(image_folder: str, target_folder: str, output_path: str) -> None:
    """
    Preprocesses the complete dataset so it can be used for training.

    Args:
        image_folder: path to images
        target_folder: path to xml files
        output_path: path to save folder
    """
    to_tensor = transforms.PILToTensor()

    target_paths = [x for x in glob.glob(f"{target_folder}/*.xml")]
    image_paths = [f"{image_folder}/{x.split(os.sep)[-1][:-4]}.jpg" for x in target_paths]

    print(f"{len(image_paths)=}")
    print(f"{len(target_paths)=}")

    for img_path, tar_path in tqdm(zip(image_paths, target_paths),
                                   total=len(image_paths),
                                   desc='preprocessing'):

        document_name = img_path.split(os.sep)[-1][:-4]
        try:
            os.makedirs(f"{output_path}/{document_name}/", exist_ok=False)
        except OSError:
            continue
        regions, mask_regions = extract(tar_path)

        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)  # type: ignore
        torch_image = to_tensor(image).permute(1, 2, 0).to(torch.uint8)

        for i, region in enumerate(regions):
            # create dict for subimage
            os.makedirs(f"{output_path}/{document_name}/region_{i}", exist_ok=True)

            # save subimage
            subimage = torch_image[region['part'][0]: region['part'][2],    # type: ignore
                                   region['part'][1]: region['part'][3]]    # type: ignore

            io.imsave(f"{output_path}/{document_name}/region_{i}/image.jpg",
                      subimage.to(torch.uint8))

            # save target information
            torch.save(region['masks'],
                       f"{output_path}/{document_name}/region_{i}/masks.pt")
            torch.save(region['baselines'],
                       f"{output_path}/{document_name}/region_{i}/baselines.pt")
            torch.save(region['bboxes'],
                       f"{output_path}/{document_name}/region_{i}/bboxes.pt")

            target = create_baseline_target(subimage.shape[:2],
                                            region['baselines'],    # type: ignore
                                            mask_regions)        # type: ignore
            np.savez_compressed(f"{output_path}/{document_name}/region_{i}/baselines",
                                array=target)


if __name__ == "__main__":
    main(f'{Path(__file__).parent.absolute()}/../../../data/images',
         f'{Path(__file__).parent.absolute()}/../../../data/pero_lines_bonn_regions',
         f'{Path(__file__).parent.absolute()}/../../../data/preprocessed')
