"""Module contains polygon conversion and export functions."""
from typing import Dict, List, Tuple

import numpy as np
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


def create_polygons(sub_mask: ndarray, label: int, tolerance: List[float], bbox_size: int) -> Tuple[
    List[List[float]], List[List[float]]]:
    """Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g., an elephant behind a tree)
    # from https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#create-custom-coco-dataset
    """
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation="low")
    segmentations: List[List[float]] = []
    bbox_list: List[List[float]] = []
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
                append_polygons(polygon, bbox_list, segmentations, bbox_size)
        else:
            append_polygons(poly, bbox_list, segmentations, bbox_size)

    return segmentations, bbox_list


def append_polygons(poly: Polygon, bbox_list: List[List[float]], segmentations: List[List[float]],
                    bbox_size: int) -> None:
    """
    Append polygon if it has at least 3 corners
    :param bbox_list: List containing bbox List with uppper left and lower right corner.
    :param poly: polygon
    :param segmentations: List containing polygons
    """
    segmentation = np.array(poly.exterior.coords).ravel().tolist()
    if len(segmentation) > 2:
        bbox = poly.bounds
        if bbox_sufficient(bbox, bbox_size):
            segmentations.append(segmentation)
            bbox_list.append(list(bbox))


def bbox_sufficient(bbox: List[float], size: int, x_axis = False) -> bool:
    """
    Calcaulates wether the edges of the bounding box are larger than parameter size. x and y edge are being summed
    up for this calculation. Eg if size = 100 x and y edges have to sum up to at least 100 Pixel.
    :param bbox: bbox list, minx, miny, maxx, maxy
    :param size: size to which the edges must at least sum to
    :x_axis: if true, only check for x axis value.
    :return: bool value wether bbox is large enough
    """
    if x_axis:
        return (bbox[2] - bbox[0]) > size
    return (bbox[2] - bbox[0]) + (bbox[3] - bbox[1]) > size


def prediction_to_polygons(pred: ndarray, tolerance: List[float], bbox_size: int) -> Tuple[
    Dict[int, List[List[float]]], Dict[int, List[List[float]]]]:
    """
    Converts prediction int ndarray to a dictionary of polygons
    :param tolerance: Array with pixel tolarance values for poygon simplification
    :param pred: prediction ndarray
    """
    masks = create_sub_masks(Image.fromarray(pred.astype(np.uint8)))
    segmentations = {}
    bbox_dict = {}
    for label, mask in masks.items():
        # debug masks
        # mask.save(f"data/output/{label}.png")
        segment, bbox = create_polygons(np.array(mask), label, tolerance, bbox_size)
        segmentations[label], bbox_dict[label] = segment, bbox
    return segmentations, bbox_dict


def get_splitting_regions(bbox_list: ndarray, big_separator_size:int) -> List[int]:
    """
    Extract big separators if they meet the threshold
    :param bbox_list: 2d n x 6 ndarray with id, label and bbox corners.
    :param big_separator_size: big separator threshold. If this requirement isnt met, the separator is too short
    :return: list of big separators
    """
    index_list = np.where(bbox_list[:, 1] == 9)[0]
    result_list = []
    for index in index_list:
        if bbox_sufficient(bbox_list[index][2:], big_separator_size, True):
            result_list.append(index)
    return result_list


def get_reading_order(bbox_list: ndarray, result: List[int], big_separator_size: int) -> None:
    """
    Calculate reading order by first separating the page by big seperators. Regions without big seperators are
    forwarded to calculate_reading_order. Big seperators are being handled separately.
    :param big_separator_size: Big separators are only considered to split the page if they exceed a certain size
    :param bbox_list: 2d n x 7 ndarray with id, label, bbox corners and y-axis center.
    :param result: Result List, is being filled over recursive calls.
    :return: list of indices in reading order
    """

    splitting_indices = get_splitting_regions(bbox_list, big_separator_size)
    if len(splitting_indices) > 0:
        splitting_index = splitting_indices[0]
        big_seperator_entry = bbox_list[splitting_index]
        bbox_list = np.delete(bbox_list, splitting_index, axis=0)

        region_bool = bbox_list[:, 6] > big_seperator_entry[3]
        calculate_reading_order(bbox_list[np.invert(region_bool)], result)
        result.append(big_seperator_entry[0])

        get_reading_order(bbox_list[region_bool], result, big_separator_size)
    else:
        calculate_reading_order(bbox_list, result)


def calculate_reading_order(bbox_list: ndarray, result: List[int]) -> None:
    """
    Receives regions without big sperators.
    Bboxes are sorted by the sum of the upper left corner to identify the upper left most element.
    Then, all elements, which begin below of that pivot element are considered one column and sorted verticly.
    This is repeated until all regions are concatenated.
    :param bbox_list:2d n x 7 ndarray with id, label, bbox corners and y-axis center.
    :param result: reading order list to append indices to
    """
    if bbox_list.size == 0:
        return
    sorted_by_sum = bbox_list[np.argsort(bbox_list[:, 2: 4].sum(axis=1))]
    while True:
        level_bool = sorted_by_sum[:, 2] <= sorted_by_sum[0, 4]
        # debug pivot elments
        # print(f"Pivot Element {len(result) + 1} with bbox {sorted_by_sum[0]}")
        current_level = sorted_by_sum[level_bool]
        current_level = current_level[np.argsort(current_level[:, 3])]

        result += list(current_level[:, 0])

        next_level = sorted_by_sum[np.invert(level_bool)]
        sorted_by_sum = next_level
        if next_level.size == 0:
            break
