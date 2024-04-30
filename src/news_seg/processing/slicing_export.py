"""Module for generating masks for slicing images and exporting the slices."""
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from numpy import ndarray


def export_slices(args: argparse.Namespace, file: str, image: ndarray,
                  reading_order_dict: Dict[int, int], segmentations: Dict[int, List[List[float]]],
                  bbox_list: Dict[int, List[List[float]]], pred: ndarray) -> None:
    """
    Cuts slices out of the input image and applies mask. Those are being saved, sorted by input
    image and reading order on that nespaper page
    :param args: arguments
    :param file: file name
    :param image: input image (c, w, h)
    :param pred: prediction
    :param reading_order_dict: Dictionary for looking up reading order
    :param segmentations: polygons
    :param pred: prediction 2d ndarray uint8
    """
    if not os.path.exists(f"{args.slices_path}{os.path.splitext(file)[0]}"):
        os.makedirs(f"{args.slices_path}{os.path.splitext(file)[0]}")

    mask_list, reading_order_list, mask_bbox_list = get_slicing(segmentations, bbox_list,
                                                                reading_order_dict,
                                                                int(args.area_size * args.scale), pred)

    reading_order_dict = {k: v for v, k in enumerate(np.argsort(np.array(reading_order_list)))}
    for index, mask in enumerate(mask_list):
        bbox = mask_bbox_list[index]
        slice_image = image[:, int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2]), ]
        mean = np.mean(slice_image, where=mask == 0)
        slice_image = slice_image * mask
        slice_image = np.transpose(slice_image, (1, 2, 0))
        slice_image[slice_image[:, :, ] == (0, 0, 0)] = mean

        Image.fromarray((slice_image * 255).astype(np.uint8)).save(
            f"{args.slices_path}{os.path.splitext(file)[0]}/{reading_order_dict[index]}.png")


def area_sufficient(bbox: List[float], size: int) -> bool:
    """
    Calcaulates wether the area of the region is larger than parameter size.
    :param bbox: bbox list, minx, miny, maxx, maxy
    :param size: size to which the edges must at least sum to
    :return: bool value wether area is large enough
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > size


def get_slicing(
        segmentations: Dict[int, List[List[float]]], bbox_list: Dict[int, List[List[float]]],
        reading_order: Dict[int, int], area_size: int, pred: ndarray
) -> Tuple[List[ndarray], List[int], List[List[float]]]:
    """
    Takes segmentation dictionary and slices it in bbox pieces for each polygon
    :param reading_order: assings reading order position to each polygon index
    :param bbox_list: Dictionaray of bboxes sorted after label
    :param shape: shape of original image
    :param segmentations: dictionary assigning labels to polygon lists
    :return: result image as ndarray, reading order list and bbox list which correspond to the chosen regions
    """
    index = 0
    masks: List[ndarray] = []
    reading_order_list: List[int] = []
    mask_bbox_list: List[List[float]] = []
    for label, segmentation in segmentations.items():
        for key, _ in enumerate(segmentation):

            bbox = bbox_list[label][key]
            if area_sufficient(bbox, area_size):
                # polygon_ndarray = np.reshape(polygon, (-1, 2)).T
                # x_coords, y_coords = draw.polygon(polygon_ndarray[1], polygon_ndarray[0])
                create_mask(bbox, index, mask_bbox_list, masks, reading_order, reading_order_list, pred)
            index += 1
    return masks, reading_order_list, mask_bbox_list


def create_mask(bbox: List[float], index: int, mask_bbox_list: List[List[float]], masks: List[ndarray],
                reading_order: Dict[int, int], reading_order_list: List[int],
                pred: ndarray) -> None:
    """
    Draw mask into empyt image and cut out the bbox area. Masks, as well as reading order and bboxes are appended to
    their respective lists for further processing
    :param bbox:
    :param index:
    :param mask_bbox_list:
    :param masks:
    :param reading_order:
    :param reading_order_list:
    :param shape:
    :param x_coords:
    :param y_coords:
    """
    # temp_image = np.zeros(shape, dtype="uint8")
    # temp_image[x_coords, y_coords] = 1
    mask = pred[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]
    mask = (mask == 4).astype(np.uint8)
    masks.append(mask)
    reading_order_list.append(reading_order[index])
    mask_bbox_list.append(bbox)
