from typing import List, Tuple

import numpy as np
from skimage import draw

LABEL_ASSIGNMENTS = {"UnknownRegion": 1, "caption": 2, "table": 3, "article": 4, "heading": 5, "header": 6,
                     "separator_vertical": 7, "separator_short": 8, "separator_horizontal": 9}


def draw_img(annotation: dict):
    """
    draws an image with the information from the read-function
    :param annotation: dict with information
    :return: ndarray
    """
    shift = 200

    x, y = annotation['size']
    img = np.zeros((y+2*shift, x+2*shift), dtype=np.uint8)

    for key, label in LABEL_ASSIGNMENTS.items():
        if key in annotation['tags'].keys():
            for polygon in annotation['tags'][key]:
                img = draw_polygon(img, polygon, label=label, shift=shift)

    return img[shift:-shift, shift:-shift]


def draw_polygon(img: np.ndarray, polygon: List[Tuple[int]], label: int = 1, shift: int = 0):
    polygon = np.array(polygon, dtype=int).T
    rr, cc = draw.polygon(polygon[1]+shift, polygon[0]+shift)
    img[rr, cc] = label

    return img