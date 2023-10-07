"""Module contains polygon conversion and export functions."""
from typing import Dict, List

import numpy as np
import torch
from numpy import ndarray
from PIL import Image
from shapely.geometry import Polygon
from skimage import measure


def create_sub_masks(mask_image: Image.Image) -> Dict[int, Image.Image]:
    """Split prediction in to submasks.
    From https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#create-custom-coco-dataset
    """
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks: Dict[int, Image] = {}
    for pos_x in range(width):
        for pos_y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((pos_x, pos_y))

            # If the pixel is not black...
            if pixel != 0:
                # Check to see if we've created a sub-mask...
                sub_mask = sub_masks.get(pixel)
                if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contour module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel] = Image.new("1", (width + 2, height + 2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel].putpixel((pos_x + 1, pos_y + 1), 1)

    return sub_masks


def create_polygons(sub_mask: ndarray, label: int, tolerance: List[float]) -> List[List[float]]:
    """Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g., an elephant behind a tree)
    # from https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#create-custom-coco-dataset
    """
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation="low")
    segmentations: List[List[float]] = []
    bbox_list = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i, coords in enumerate(contour):
            row, col = coords
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(tolerance[label - 1], preserve_topology=False)
        if poly.geom_type == 'MultiPolygon':
            multi_polygons = list(poly.geoms)
            for polygon in multi_polygons:
                append_polygons(polygon, segmentations)
        else:
            append_polygons(poly, segmentations)

    return segmentations, bbox_list


def append_polygons(poly: Polygon, bbox_list: List[Tuple[int, ...]], segmentations: List[List[float]]) -> None:
    """
    Append polygon if it has at least 3 corners
    :param poly: polygon
    :param segmentations: list to append
    """
    segmentation = np.array(poly.exterior.coords).ravel().tolist()
    if len(segmentation) > 2:
        segmentations.append(segmentation)
        bbox_list.append(poly.bounds)


def prediction_to_polygons(pred: ndarray, tolerance: List[float]) -> Dict[int, List[List[float]]]:
    """
    Converts prediction int ndarray to a dictionary of polygons
    :param pred:
    """
    masks = create_sub_masks(Image.fromarray(pred.astype(np.uint8)))
    segmentations = {}
    bbox_dict = {}
    for label, mask in masks.items():
        segmentations[label], bbox_dict[label] = create_polygons(np.array(mask), label, tolerance)
    return segmentations, bbox_dict


def get_reading_order(bbox_list: ndarray) -> List[int]:
    """
    Calculate reading order by first seperating regions by big seperators. Regions without big seperators are
    forwarded to calculate_reading_order. Big seperators are being handelt seperately.
    :param bbox_list: 2d n x 5 ndarray with id, label and bbox corners.
    :return: list of indices in reading order
    """
    result = np.array([])
    big_seperator_indices = np.where(bbox_list[bbox_list[:, 1] == 9])
    if big_seperator_indices:
        region_bool = bbox_list[:, 3] > bbox_list[big_seperator_indices[0], 3]

        calculate_reading_order(np.delete(bbox_list[region_bool.invert()], big_seperator_indices[0]), result)
        result.append(bbox_list[big_seperator_indices[0]])

        get_reading_order(bbox_list[region_bool])
    else:
        calculate_reading_order(bbox_list, result)
    return result[:, 0].tolist()


def calculate_reading_order(bbox_list: ndarray, result: ndarray) -> None:
    """
    Receives regions without big sperators.
    Bboxes are sorted by the sum of the upper left corner to identify the upper left most element.
    Then, all elements, which begin below of the first element are considered one column and sorted verticly.
    This is repeated until all regions are concatenated.
    :param bbox_list:
    :param result:
    """
    sorted_by_sum = bbox_list[:, np.argsort(bbox_list[:, 3: 5].sum(axis=0))[::-1]]
    while True:
        level_bool = np.where(sorted_by_sum[:, 2] <= sorted_by_sum[0, 4])
        current_level = sorted_by_sum[level_bool]
        current_level = current_level[:, np.argsort(bbox_list[:, 3])[::-1]]

        result = np.concatenate((result, current_level), axis=0)

        next_level = sorted_by_sum[level_bool.invert()]
        current_level = next_level
        if not next_level:
            break
