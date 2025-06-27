"""
Module for drawing polygon data into images.
"""

from typing import Dict, List, Tuple, Any

import numpy as np
from numpy import ndarray
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from skimage import draw

from cgprocess.layout_segmentation.class_config import PAGE_BORDER_CONTENT
from cgprocess.layout_segmentation.class_config import LABEL_ASSIGNMENTS, PADDING_LABEL


def draw_img(annotation: dict, mark_page_border: bool = False) -> ndarray:
    """
    draws an image with the information from the transkribus xml read function. If activates, the outside areas of
    the image, that do not contain polygons are marked, such that they can be ignored.
    image
    :param annotation: dict with information
    :param mark_page_border: Activate page border marking.
    :return: ndarray
    """
    shift = 200

    x_size, y_size = annotation["size"]
    img = np.zeros((y_size + 2 * shift, x_size + 2 * shift), dtype=np.uint8)

    # then draw regions in order if label equals zero, the region is skipped, because it would draw zeros in an image
    # initialized with zeros.
    polygon_list = []
    for key, label in LABEL_ASSIGNMENTS.items():
        if label != 0 and key in annotation["tags"]:
            img = process_polygons(annotation, img, key, label, mark_page_border, polygon_list, shift)

    if mark_page_border and len(polygon_list) > 0:
        combined_polygon = unary_union(polygon_list)
        if combined_polygon.geom_type == "MultiPolygon":
            points = np.vstack([np.array(element.exterior.coords) for element in combined_polygon.geoms])
        else:
            points = np.array(combined_polygon.exterior.coords)
        image_bounds = box(0,0,img.shape[1]-1, img.shape[0]-1)
        margin = int(min(img.shape[0],img.shape[1]) * 0.02)
        hull = ConvexHull(points)
        hull_points = (Polygon(points[hull.vertices]).buffer(margin).intersection(image_bounds)).exterior.coords
        draw_polygon(img, hull_points, label=PADDING_LABEL, shift=shift, invert=True)
        if PAGE_BORDER_CONTENT in annotation["tags"]:
            process_polygons(annotation, img, PAGE_BORDER_CONTENT, 0, False, polygon_list, shift)

    return img[shift:-shift, shift:-shift]


def process_polygons(annotation: dict, img: ndarray, key: str, label: int, mark_page_border: bool,
                     polygon_list: list, shift: int) -> ndarray:
    """
    Call polygon draw function and fill polygon list for all polygons of this key.
    Returns: ndarray img with polygons drawn.
    """
    for polygon in annotation["tags"][key]:
        img = draw_polygon(img, polygon, label=label, shift=shift)
        if mark_page_border:
            polygon_list.append(Polygon(polygon).buffer(0))
    return img


def draw_polygon(
        img: ndarray, polygon: List[Tuple[str]], label: int = 1, shift: int = 0, invert: bool = False
) -> ndarray:
    """Takes corner coordinates and fills entire polygon with label values"""
    polygon_np = np.array(polygon, dtype=int).T
    x_coords, y_coords = draw.polygon(polygon_np[1] + shift, polygon_np[0] + shift)
    if invert:
        print(img.shape)
        mask = np.ones_like(img, dtype=bool)
        mask[x_coords, y_coords] = False
        img[mask] = label
    else:
        img[x_coords, y_coords] = label
    return img


def draw_polygons_into_image(
        segmentations: Dict[int, List[List[float]]], shape: Tuple[int, ...]
) -> ndarray:
    """
    Takes segmentation dictionary and draws polygons with assigned labels into a new image.
    :param shape: shape of original image
    :param segmentations: dictionary assigning labels to polygon lists
    :return: result image as ndarray
    """

    polygon_pred = np.zeros(shape, dtype="uint8")
    for label, segmentation in segmentations.items():
        for polygon in segmentation:
            polygon_ndarray = np.reshape(polygon, (-1, 2)).T
            x_coords, y_coords = draw.polygon(polygon_ndarray[1], polygon_ndarray[0])
            polygon_pred[x_coords, y_coords] = label
    return polygon_pred
