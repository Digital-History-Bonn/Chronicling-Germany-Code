"""Preprocess for the Newspaper dataset to predict Textlines."""

import argparse
import glob
import os
from multiprocessing import Process, Queue, set_start_method
from time import sleep
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon
from shapely.ops import split
from skimage import draw
from torchvision import transforms
from tqdm import tqdm

from cgprocess.baseline_detection.utils import adjust_path, extract


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
    new_coords = (
        [(-100, points[0][1])] + points + [(points[-1][0] + 100, points[-1][1])]
    )
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

    raise ValueError("Baseline and polygone not intersecting!")


def calc_heights(
    polygon: Polygon, baseline: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        rr, cc = draw.line(
            baseline[i, 0], baseline[i, 1], baseline[i + 1, 0], baseline[i + 1, 1]
        )
        x_coords.extend(rr)
        y_coords.extend(cc)

    baseline_x = torch.tensor(x_coords)
    baseline_y = torch.tensor(y_coords)

    indices = (
        (baseline_y - torch.tensor([bbox[0]])).long().clip(min=0, max=shape[1] - 1)
    )
    values = torch.sum(polygon_image, dim=0)[indices]

    return baseline_x, baseline_y, values


def draw_baseline_target(
    shape: Tuple[int, int],
    baselines: List[torch.Tensor],
    textlines: List[torch.Tensor],
    mask_regions: List[torch.Tensor],
    textregions: List[torch.Tensor],
    name: str,
    width: int = 3,
) -> np.ndarray:
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
        try:
            line = LineString(torch.flip(baseline, dims=[1]))
            polygon = Polygon(torch.flip(textline, dims=[1]))
        except ValueError as e:
            print(f"{e} in image {name}\n")
            continue

        # draw limiter
        draw_limiter(limiter_draw, polygon, width)

        # draw baseline
        baseline_draw.line(line.coords, fill=1, width=1)

        # draw ascender/descender
        try:
            ascender, descender = split_textbox(polygon, line)
            draw_ascender_descender(ascender, baseline, shape, target, dim=0)
            draw_ascender_descender(descender, baseline, shape, target, dim=1)
        except (ValueError, IndexError) as e:
            print(f"{e} in image {name}\n")
            continue

    target[:, :, 3] = np.array(limiter_img)
    target[:, :, 2] = np.array(baseline_img)

    # draw textregions
    for textregion in textregions:
        textregion_draw.polygon(
            [(int(x[1]), int(x[0])) for x in textregion], fill=0, outline=1, width=width
        )

    target[:, :, 4] = np.array(textregion_img)

    # draw masks
    for mask_region in mask_regions:
        if len(mask_region) >= 3:
            rr, cc = draw.polygon(mask_region[:, 0], mask_region[:, 1], shape=shape)
            target[rr, cc, 5] = 0

    return np.array(target)


def draw_limiter(
    limiter_draw: ImageDraw.ImageDraw, polygon: Polygon, width: int
) -> None:
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


def draw_ascender_descender(
    polygon: Polygon,
    baseline: torch.Tensor,
    shape: Tuple[int, int],
    target: np.ndarray,
    dim: int,
) -> None:
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


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="preprocess")
    parser.add_argument(
        "--image_dir",
        "-i",
        type=str,
        default=None,
        help="path for folder with images. Images need to be jpg.",
    )

    parser.add_argument(
        "--annotation_dir",
        "-a",
        type=str,
        default=None,
        help="path for folder with layout xml files.",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="path to the folder where to save the preprocessed trainings targets",
    )

    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=1,
        help="number of processes that are used.",
    )

    return parser.parse_args()


def main() -> None:
    """Preprocesses the complete dataset so it can be used for training."""

    # get args
    args = get_args()
    target_folder = adjust_path(args.annotation_dir)
    image_folder = adjust_path(args.image_dir)
    output_path = adjust_path(args.output_dir)

    # check args
    if target_folder is None:
        raise ValueError("Please enter a valid path to annotation data!")

    if image_folder is None:
        raise ValueError("Please enter a valid path to image data!")

    if output_path is None:
        raise ValueError("Please enter a valid path to output folder!")

    os.makedirs(output_path, exist_ok=True)
    to_tensor = transforms.PILToTensor()

    target_paths = list(glob.glob(f"{target_folder}/*.xml"))
    image_paths = [
        f"{image_folder}/{x.split(os.sep)[-1][:-4]}.jpg" for x in target_paths
    ]

    print(f"{len(image_paths)=}")
    print(f"{len(target_paths)=}")

    path_queue: Queue = Queue()
    for img_path, tar_path in tqdm(
        zip(image_paths, target_paths), total=len(image_paths), desc="put in queue"
    ):
        document_name = img_path.split(os.sep)[-1][:-4]

        # check if folder already exists if yes skip this image
        if os.path.exists(f"{output_path}/{document_name}.npz"):
            continue
        path_queue.put(
            (document_name, img_path, output_path, tar_path, to_tensor, False)
        )

    processes = [
        Process(target=preprocess, args=(path_queue, i)) for i in range(args.processes)
    ]
    for process in processes:
        process.start()
    total = path_queue.qsize()
    # pylint: disable=duplicate-code
    with tqdm(total=total, desc="Preprocess", unit="pages") as pbar:
        while not path_queue.empty():
            pbar.n = total - path_queue.qsize()
            pbar.refresh()
            sleep(1)
    for _ in processes:
        path_queue.put(("", "", "", "", "", True))
    for process in tqdm(processes, desc="Waiting for processes to end"):
        process.join()


def preprocess(path_queue: Queue, i: int) -> None:
    """Get Paths from Queue and save data after preprocessing."""
    print(f"process {i}")
    while True:
        document_name, _, output_path, tar_path, _, done = path_queue.get()
        if done:
            break
        print(f"process {i} start page")
        # extract annotation information from annotation xml file
        regions, mask_regions, shape = extract(tar_path)
        # create a list for masks, baselines, bboxes and textregion information
        masks: List[torch.Tensor] = []
        baselines: List[torch.Tensor] = []
        bboxes: List[torch.Tensor] = []
        textregion: List[torch.Tensor] = []
        for region in regions:
            # save target information
            masks.extend(region["textline_polygone"])  # type: ignore
            baselines.extend(region["baselines"])  # type: ignore
            bboxes.extend(region["bboxes"])  # type: ignore
            textregion.append(region["textregion"])  # type: ignore
        # create target as numpy array and save it in a compressed file
        target = draw_baseline_target(
            shape, baselines, masks, mask_regions, textregion, document_name
        )
        np.savez_compressed(f"{output_path}/{document_name}", array=target)


if __name__ == "__main__":
    set_start_method("spawn")
    main()
