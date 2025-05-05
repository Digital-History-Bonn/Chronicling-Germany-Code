"""Module for polygon conversion"""

from typing import Dict, List, Tuple

import cv2
import numpy as np
from numpy import ndarray
from PIL import Image
from shapely.geometry import Polygon
from skimage import measure


def create_label_masks(pred: ndarray) -> Dict[int, Image.Image]:
    """Split prediction in to submasks. Creates a submask for each unique value except 0.
    Numpy implementation of python version from
     https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#create-custom-coco-dataset
    """

    unique_values = np.unique(pred)
    if 0 in unique_values:
        unique_values = np.delete(unique_values, np.where(unique_values == 0)[0][0])

    sub_masks: Dict[int, Image.Image] = {}

    for value in unique_values:
        sub_masks[value] = Image.fromarray(np.pad(pred == value, 1))

    return sub_masks


def create_mask_polygons(
    sub_mask: ndarray, label: int, tolerance: List[float], bbox_size: int, export: bool
) -> Tuple[List[List[float]], List[List[float]]]:
    """Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g., an elephant behind a tree)
    # from https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#create-custom-coco-dataset
    :param sub_mask: current submask, from that polygons will be calculated
    :param label: current class label
    :param tolerance: tolerance array which contains pixel toplerance for simplifying polygons
    :param bbox_size: size to which the edges must at least sum to
    :param export: wheter transkribus export or output_path is activated.
    If this is not the case, polygon simplification ist not necessary.
    :return:
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
        if export:
            poly = poly.simplify(tolerance[label - 1], preserve_topology=False)
        if poly.geom_type == "MultiPolygon":
            # pylint: disable=no-member
            multi_polygons = list(poly.geoms)
            for polygon in multi_polygons:
                append_polygons(polygon, bbox_list, segmentations, bbox_size)
        else:
            append_polygons(poly, bbox_list, segmentations, bbox_size)

    return segmentations, bbox_list


def append_polygons(
    poly: Polygon,
    bbox_list: List[List[float]],
    segmentations: List[List[float]],
    bbox_size: int,
) -> None:
    """
    Append polygon if it has at least 3 corners
    :param bbox_list: List containing bbox List with uppper left and lower right corner.
    :param poly: polygon
    :param segmentations: List containing polygons
    :param bbox_size: size to which the edges must at least sum to
    """
    segmentation = np.array(poly.exterior.coords).ravel().tolist()
    if len(segmentation) > 2:
        bbox = poly.bounds
        if bbox_sufficient(bbox, bbox_size):
            segmentations.append(segmentation)
            bbox_list.append(list(bbox))


def bbox_sufficient(bbox: List[float], size: int, x_axis: bool = False) -> bool:
    """
    Calcaulates wether the edges of the bounding box are larger than parameter size. x and y edge are being summed
    up for this calculation. Eg if size = 100 x and y edges have to sum up to at least 100 Pixel.
    :param x_axis: if true, only check for x axis value.
    :param bbox: bbox list, minx, miny, maxx, maxy
    :param size: size to which the edges must at least sum to
    :return: bool value wether bbox is large enough
    """
    if x_axis:
        return (bbox[2] - bbox[0]) > size
    return (bbox[2] - bbox[0]) + (bbox[3] - bbox[1]) > size


def prediction_to_region_polygons(
    pred: ndarray, tolerance: List[float], bbox_size: int, export: bool
) -> Tuple[Dict[int, List[List[float]]], Dict[int, List[List[float]]]]:
    """
    Converts prediction int ndarray to a dictionary of polygons by splitting the prediction into binary label masks.
    For each mask, polygons are created and appended to a list.
    :param tolerance: Array with pixel tolarance values for poygon simplification
    :param pred: prediction ndarray
    :param export: wheter transkribus export or output_path is activated.
        If this is not the case, only the article and horizontal seperator class bboxes are of relevance.
        Everything else will be skipped to improve performance.
    """
    masks = create_label_masks(pred)

    segmentations = {}
    bbox_dict = {}
    for label, mask in masks.items():
        # TODO: fix this, by introducing a class variable for classes excluded from polygon extraction.
        if export or label == 3 and not label in [1, 5, 8, 9, 2]:
            # for sclice export only paragraph and separator are processed (obsolete)
            segment, bbox = create_mask_polygons(
                np.array(mask), label, tolerance, bbox_size, export
            )
            segmentations[label], bbox_dict[label] = segment, bbox

    return segmentations, bbox_dict


def uncertainty_to_polygons(
    pred: ndarray,
) -> Tuple[Dict[int, List[List[float]]], Dict[int, List[List[float]]]]:
    """
    Converts the uncertain pixel image into polygones
    :param pred: map of uncertaion pixels ndarray [B, C, H, W]
    return: dict with segmentations and dict with bboxes
    """
    segmentations: Dict[int, List[List[float]]] = {i: [] for i in range(10)}
    bbox_dict: Dict[int, List[List[float]]] = {i: [] for i in range(10)}

    # pylint: disable=unpacking-non-sequence
    contours, hierarchy = cv2.findContours(
        pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )  # pylint: disable=no-member

    # calc inner and outer contours
    inner_outer = []
    for item in hierarchy[0, :, 3]:
        if item == -1:
            inner_outer.append(True)
        else:
            inner_outer.append(not inner_outer[item])

    for contour, inner in zip(contours, inner_outer):
        if len(contour) <= 3:
            continue
        contour = contour.squeeze()
        segmentations[1 if inner else 2].append(list(contour.flatten()))
        bbox_dict[1 if inner else 2].append(
            [
                contour[:, 0].max(),
                contour[:, 1].min(),
                contour[:, 0].min(),
                contour[:, 1].max(),
            ]
        )

    return segmentations, bbox_dict
