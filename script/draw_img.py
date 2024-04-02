"""
Module for converting region data into an image containing label values for each pixel.
"""
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from skimage import draw

# The order dictates the priority in the drawing process. Eg. "image": 10 assigns label 10 to image regions, but the
# drawn region will be overwritten by tables, which are further down the dictionary.
LABEL_ASSIGNMENTS = {
    "TextLine": 0,
    "UnknownRegion": 1,
    "image": 10,
    "inverted_text": 11,
    "caption": 2,
    "table": 3,
    "article": 4,
    "article_": 4,
    "heading": 5,
    "header": 6,
    "separator_fancy": 7,
    "separator_vertical": 7,
    "separator_short": 8,
    "separator_horizontal": 9,
}

LABEL_NAMES = [
    "UnknownRegion",
    "caption",
    "table",
    "article",
    "heading",
    "header",
    "separator_vertical",
    "separator_short",
    "separator_horizontal",
    "image",
    "inverted_text",
]


def draw_img(annotation: dict) -> ndarray:
    """
    draws an image with the information from the read-function
    :param annotation: dict with information
    :return: ndarray
    """
    shift = 200

    x_size, y_size = annotation["size"]
    img = np.zeros((y_size + 2 * shift, x_size + 2 * shift), dtype=np.uint8)

    # first draw all unkown regions
    for key, polygons in annotation["tags"].items():
        if key not in LABEL_ASSIGNMENTS:
            for polygon in polygons:
                if len(polygon) > 0:
                    img = draw_polygon(img, polygon, shift=shift)

    # then draw regions in order if label equals zero, the region is skipped, because it would draw zeros in an image
    # initialized with zeros.
    for key, label in LABEL_ASSIGNMENTS.items():
        if label != 0 and key in annotation["tags"]:
            for polygon in annotation["tags"][key]:
                img = draw_polygon(img, polygon, label=label, shift=shift)

    return img[shift:-shift, shift:-shift]


def draw_polygon(
        img: ndarray, polygon: List[Tuple[int]], label: int = 1, shift: int = 0
) -> ndarray:
    """Takes corner coordinates and fills entire polygon with label values"""
    polygon_np = np.array(polygon, dtype=int).T
    x_coords, y_coords = draw.polygon(polygon_np[1] + shift, polygon_np[0] + shift)
    img[x_coords, y_coords] = label

    return img
