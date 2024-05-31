"""Utility functions for OCR."""
import random
from typing import Tuple, Union, Optional

import numpy as np
import torch
from PIL import Image, ImageOps
from bs4 import BeautifulSoup, PageElement
from matplotlib import pyplot as plt


def pad_xml(soup: BeautifulSoup, pad_value: int = 10) -> BeautifulSoup:
    """
    Pads the coordinates of all elements in given BeautifulSoup soup.

    Args:
        soup: BeautifulSoup with annotations.
        pad_value: Value for padding

    Returns:
        Padded BeautifulSoup soup
    """
    # Find all elements with 'points' attribute
    elements_with_points = soup.find_all(attrs={"points": True})

    for element in elements_with_points:
        points = element['points']
        padded_points = pad_points(points, pad_value)
        element['points'] = padded_points

    return soup


def pad_points(points: str, pad_value: int = 10) -> str:
    """
    Pads each point in the points string by the given pad value.

    Args:
        points: A string of points in the format 'x1,y1 x2,y2 ...'.
        pad_value: The padding value (int) to apply to each point.

    Returns:
        A string of padded points.
    """
    points_list = points.split()
    padded_points_list = []

    for point in points_list:
        x, y = map(int, point.split(','))
        padded_x = x + pad_value
        padded_y = y + pad_value
        padded_points_list.append(f"{padded_x},{padded_y}")

    return ' '.join(padded_points_list)


def pad_image(image: Image.Image, pad: int = 10, color: tuple = (0, 0, 0)) -> Image.Image:
    """
    Pads the given PIL Image with a specified number of pixels on each side.

    Args:
        image (Image.Image): The input PIL Image to be padded.
        pad (int): The number of pixels to pad on each side. Default is 10.
        color (tuple): The color for the padding. Default is white (255, 255, 255).

    Returns:
        Image.Image: The padded PIL Image.
    """
    return ImageOps.expand(image, border=(pad, pad, pad, pad), fill=color)


def get_bbox(points: Union[np.ndarray, torch.Tensor],  # type: ignore
             ) -> Tuple[int, int, int, int]:
    """
    Creates a bounding box around all given points.

    Args:
        points: p.ndarray of shape (N x 2) containing a list of points

    Returns:
        coordinates of bounding box in the format (x min, y_min, x_max, y_mx)
    """
    x_max, x_min = points[:, 0].max(), points[:, 0].min()
    y_max, y_min = points[:, 1].max(), points[:, 1].min()
    return x_min, y_min, x_max, y_max  # type: ignore


def convert_coord(element: PageElement) -> np.ndarray:
    """
    Converts PageEZement with Coords in to a numpy array.

    Args:
        element: PageElement with Coords for example a Textregion

    Returns:
        np.ndarray of shape (N x 2) containing a list of coordinates
    """
    coords = element.find('Coords')
    return np.array([tuple(map(int, point.split(','))) for
                     point in coords['points'].split()])[:, np.array([1, 0])]


def plot_boxes_on_image(image: Image.Image, baselines: torch.Tensor, polygons: torch.Tensor,
                        name: str) -> None:
    """
    Plots baselines and textline polygons on image.

    Args:
        image: image to be plotted.
        baselines: baseline coordinates.
        polygons: textline coordinates.
        name: name of the file to be saved.
    """
    # Create figure and axes
    _, ax = plt.subplots()

    # Display the image
    ax.imshow(image, cmap='gray')

    for baseline in baselines:
        ax.plot(baseline[:, 0], baseline[:, 1], color='blue', linewidth=.02)

    for polygon in polygons:
        ax.plot(polygon[:, 0], polygon[:, 1], color='orange', linewidth=.02)

    # save plot
    plt.savefig(f'{name}.png', dpi=750)


def adjust_path(path: Optional[str]) -> Optional[str]:
    """
    Make sure, there is a slash at the end of a (folder) spath string.

    Args:
        path: String representation of path

    Returns:
        path without ending '/'
    """
    return path if not path or path[-1] != '/' else path[:-1]


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
