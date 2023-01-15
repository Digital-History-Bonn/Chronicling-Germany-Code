"""
Module for converting region data into an image containing label values for each pixel.
"""
from typing import List, Tuple

import numpy as np
from skimage import draw  # type: ignore

LABEL_ASSIGNMENTS = {"UnknownRegion": 1, "caption": 2, "table": 3, "article": 4, "article_": 4, "heading": 5,
                     "header": 6,
                     "separator_vertical": 7, "separator_short": 8, "separator_horizontal": 9}


def draw_img(annotation: dict):
    """
    draws an image with the information from the read-function
    :param annotation: dict with information
    :return: ndarray
    """
    shift = 200

    x_size, y_size = annotation['size']
    img = np.zeros((y_size + 2 * shift, x_size + 2 * shift), dtype=np.uint8)

    for key, label in LABEL_ASSIGNMENTS.items():
        if key in annotation['tags'].keys():
            for polygon in annotation['tags'][key]:
                img = draw_polygon(img, polygon, label=label, shift=shift)

    return img[shift:-shift, shift:-shift]


def draw_polygon(img: np.ndarray, polygon: List[Tuple[int]], label: int = 1, shift: int = 0) -> np.ndarray:
    """Takes corner coordinates and fills entire polygon with label values"""
    polygon_np = np.array(polygon, dtype=int).T  # type: ignore
    x_coords, y_coords = draw.polygon(polygon_np[1] + shift, polygon_np[0] + shift)
    img[x_coords, y_coords] = label

    return img
