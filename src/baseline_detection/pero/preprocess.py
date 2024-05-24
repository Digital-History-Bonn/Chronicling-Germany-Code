"""Preprocess for the Newspaper dataset to predict Textlines."""

import glob
import os
from pathlib import Path
from typing import List, Tuple

import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from skimage import draw
from matplotlib import pyplot as plt
from shapely.ops import split
from shapely.geometry import LineString, Polygon
from tqdm import tqdm

from src.baseline_detection.utils import extract


def split_textbox(textline: Polygon, baseline: LineString) -> Tuple[Polygon, Polygon]:
    """
    Splits textline polygone into ascender and descender by separating with the baseline.

    Args:
        textline: shapely Polygon of textline
        baseline: shapely LineString of baseline

    Raises:
        ValueError: If baseline is not intersecting the given textline

    Returns:
        ascender, descender
    """
    # extend line to avoid not completely intersecting polygone
    points = list(baseline.coords)
    new_coords = [(-100, points[0][1])] + points + [(points[-1][0] + 100, points[-1][1])]
    baseline = LineString(new_coords)

    # Use the split method to split the polygon
    parts = split(textline, baseline)

    if len(parts.geoms) >= 2:
        # Determine which part is above and below the split line
        if parts.geoms[0].centroid.y < parts.geoms[1].centroid.y:
            ascender = parts.geoms[0]
            descender = parts.geoms[1]
        else:
            ascender = parts.geoms[1]
            descender = parts.geoms[0]
        return ascender, descender

    raise ValueError('Baseline and polygone not intersecting!')


def calc_heights(polygon: Polygon,
                 baseline: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the heights of the polygon at all x coordinates.

    Args:
        polygon: shapely Polygon of ascender or descender
        baseline: shapely LineString of baseline

    Returns:
        x and y coordinates of the baseline and the corresponding height values
    """
    bbox = polygon.bounds

    # Move the coordinates relative to the bottom-left corner of its bounding box
    relative_polygon = np.array(polygon.exterior.coords) - np.array(bbox[:2])

    # calc width and hight of polygon image
    shape = (int(bbox[3] - bbox[1]), int(bbox[2] - bbox[0]))

    # draw polygon
    polygon_image = torch.zeros(size=shape, dtype=torch.int)
    rr, cc = draw.polygon(relative_polygon[:, 1], relative_polygon[:, 0], shape=shape)
    polygon_image[rr, cc] = 1

    x_coords = []
    y_coords = []
    for i in range(len(baseline) - 1):
        rr, cc = draw.line(baseline[i, 0],
                           baseline[i, 1],
                           baseline[i + 1, 0],
                           baseline[i + 1, 1])
        x_coords.extend(rr)
        y_coords.extend(cc)

    baseline_x = torch.tensor(x_coords)
    baseline_y = torch.tensor(y_coords)

    indices = (baseline_y - torch.tensor([bbox[0]])).long().clip(min=0, max=shape[1] - 1)
    values = torch.sum(polygon_image, dim=0)[indices]

    return baseline_x, baseline_y, values


def draw_baseline_target(shape: Tuple[int, int],
                         baselines: List[torch.Tensor],
                         textlines: List[torch.Tensor],
                         mask_regions: List[torch.Tensor],
                         textregions: List[torch.Tensor],
                         name: str,
                         width: int = 3) -> np.ndarray:
    """
    Draw baseline target for given shape.

    Args:
        shape: shape of target
        baselines: list of baselines
        textlines: list of polygons around the textline
        mask_regions: list of polygons to mask on image
        textregions: list of polygons around paragraph
        name: filename for logging errors
        width: width of the drawn baselines

    Returns:
        np.array with targets
    """
    # Create a blank target filled with ones (white)
    target = np.zeros((*shape, 6), dtype=np.uint8)
    target[:, :, 5] = 1  # mask with default value true

    # create PILDraw instances to draw baselines, textregions and limiters
    limiter_img = Image.fromarray(target[:, :, 3])
    limiter_draw = ImageDraw.Draw(limiter_img)

    textregion_img = Image.fromarray(target[:, :, 4])
    textregion_draw = ImageDraw.Draw(textregion_img)

    baseline_img = Image.fromarray(target[:, :, 2])
    baseline_draw = ImageDraw.Draw(baseline_img)

    # Draw targets
    for baseline, textline in zip(baselines, textlines):
        line = LineString(torch.flip(baseline, dims=[1]))
        polygon = Polygon(torch.flip(textline, dims=[1]))

        # draw limiter
        draw_limiter(limiter_draw, polygon, width)

        # draw baseline
        baseline_draw.line(line.coords, fill=1, width=1)

        try:
            ascender, descender = split_textbox(polygon, line)
        except ValueError as e:
            print(f"{e} in image {name}\n")
            continue
        # draw ascender/descender
        draw_ascender_descender(ascender, baseline, shape, target, dim=0)
        draw_ascender_descender(descender, baseline, shape, target, dim=1)

    target[:, :, 3] = np.array(limiter_img)
    target[:, :, 2] = np.array(baseline_img)

    # draw textregions
    for textregion in textregions:
        textregion_draw.polygon([(int(x[1]), int(x[0])) for x in textregion],
                                fill=0,
                                outline=1,
                                width=width)

    target[:, :, 4] = np.array(textregion_img)

    # draw masks
    for mask_region in mask_regions:
        if len(mask_region) >= 3:
            rr, cc = draw.polygon(mask_region[:, 0], mask_region[:, 1], shape=shape)
            target[rr, cc, 5] = 0

    return np.array(target)


def draw_limiter(limiter_draw: ImageDraw, polygon: Polygon, width: int):
    """
    Draw limiter on given polygon.

    Args:
        limiter_draw: ImageDraw object to draw limiter on
        polygon: shapely Polygon representing the textline polygon
        width: width of the drawn limiter
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    limiter_draw.line([(min_x, min_y), (min_x, max_y)], fill=1, width=width)
    limiter_draw.line([(max_x, min_y), (max_x, max_y)], fill=1, width=width)


def draw_ascender_descender(polygon: Polygon,
                            baseline: torch.Tensor,
                            shape: Tuple[int, int],
                            target: np.ndarray,
                            dim: int):
    """
    Draws the given ascender oder descender as height values on the baseline positions.

    Args:
        polygon: shapely Polygon of ascender or descender
        baseline: torch Tensor representing the baseline
        shape: shape of target
        target: np array representing the target
        dim: dimension where to draw ascender/descender in target
    """
    x_coords, y_coords, values = calc_heights(polygon, baseline)
    x_coords = x_coords[y_coords < shape[1]]
    values = values[y_coords < shape[1]]
    y_coords = y_coords[y_coords < shape[1]]
    target[x_coords, y_coords, dim] = values


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


def plot_target(image: np.ndarray,
                target: np.ndarray,
                figsize: Tuple[int, int] = (50, 10),
                dpi: int = 500) -> None:
    """
    Creates and saves an Image of the targets on the image.

    Args:
        image: numpy array representation of the image (width, height, channel)
        target: numpy array representation of the target (width, height, channel)
        figsize: size of the plot (default: 50, 10)
        dpi: dpi of the plot
    """
    # Plot the first image
    attributes = ['ascenders', 'descenders', 'baselines', 'marker', 'textregion']
    image = image * target[:, :, 5, None]

    _, axes = plt.subplots(1, len(attributes), figsize=figsize)

    print(f"{target.shape=}")
    for i, attribute in enumerate(attributes):
        axes[i].imshow(image.astype(np.uint8))
        if attribute in ['ascenders', 'descenders']:
            axes[i].imshow(target[:, :, i].astype(np.uint8) * 4, alpha=0.5)
        else:
            axes[i].imshow(target[:, :, i].astype(np.uint8), cmap='gray', alpha=0.5)
        axes[i].set_title(attribute, fontsize=26)
        axes[i].axis('off')

    plt.tight_layout()  # Adjust layout
    plt.subplots_adjust(wspace=0.05)  # Adjust space between subplots

    # Display the plot with higher DPI
    plt.savefig(f'{Path(__file__).parent.absolute()}/../../../data/assets/TargetExample.png',
                dpi=dpi)
    plt.show()


def main(image_folder: str, target_folder: str, output_path: str) -> None:
    """
    Preprocesses the complete dataset so it can be used for training.

    Args:
        image_folder: path to images
        target_folder: path to xml files
        output_path: path to save folder
    """
    to_tensor = transforms.PILToTensor()

    target_paths = list(glob.glob(f"{target_folder}/*.xml"))
    image_paths = [f"{image_folder}/{x.split(os.sep)[-1][:-4]}.jpg" for x in target_paths]

    print(f"{len(image_paths)=}")
    print(f"{len(target_paths)=}")

    for img_path, tar_path in tqdm(zip(image_paths, target_paths),
                                   total=len(image_paths),
                                   desc='preprocessing'):
        document_name = img_path.split(os.sep)[-1][:-4]

        # check if folder already exists if yes skip this image
        if os.path.exists(f"{output_path}/target_{document_name}.npz"):
            continue

        # extract annotation information from annotation xml file
        regions, mask_regions = extract(tar_path)

        # open image, check orientation, convert to tensor
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)  # type: ignore
        torch_image = to_tensor(image).permute(1, 2, 0).to(torch.uint8)

        # create a list for masks, baselines, bboxes and textregion information
        masks: List[torch.Tensor] = []
        baselines: List[torch.Tensor] = []
        bboxes: List[torch.Tensor] = []
        textregion: List[torch.Tensor] = []
        for region in regions:
            # save target information
            masks.extend(region['textline_polygone'])
            baselines.extend(region['baselines'])
            bboxes.extend(region['bboxes'])
            textregion.append(region['textregion'])     # type: ignore

        # create target as numpy array and save it in a compressed file
        target = draw_baseline_target(torch_image.shape[:2],
                                      baselines,
                                      masks,
                                      mask_regions,
                                      textregion,
                                      document_name)
        np.savez_compressed(f"{output_path}/{document_name}", array=target)


if __name__ == "__main__":
    main(f'{Path(__file__).parent.absolute()}/../../../data/images',
         f'{Path(__file__).parent.absolute()}/../../../data/pero_lines_bonn_regions',
         f'{Path(__file__).parent.absolute()}/../../../data/preprocessed')

    # test_array = np.load(
    #     f'{Path(__file__).parent.absolute()}/../../'
    #     f'data/Newspaper/preprocessed3/'
    #     f'Koelnische Zeitung 1866.06-1866.09 - 0182/baselines.npz')['array']
    # image = io.imread(
    #     f'{Path(__file__).parent.absolute()}/../../'
    #     f'data/Newspaper/newspaper-dataset-main-images/'
    #     f'images/Koelnische Zeitung 1866.06-1866.09 - 0182.jpg')
    # plot_target(image, test_array)
