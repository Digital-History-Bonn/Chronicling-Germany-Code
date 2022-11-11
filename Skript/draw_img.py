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

    x, y = annotation['size']
    img = np.zeros((x, y), dtype=int)

    for key, label in LABEL_ASSIGNMENTS.items():
        if key in annotation['tags'].keys():
            for polygon in annotation['tags'][key]:
                img = draw_polygon(img, polygon, label=label)

    return img


def draw_polygon(img: np.ndarray, polygon: List[Tuple[int]], label: int = 1):
    polygon = np.array(polygon, dtype=int).T
    rr, cc = draw.polygon(polygon[1], polygon[0])
    img[rr, cc] = label

    return img