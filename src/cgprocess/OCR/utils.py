"""Utility functions for OCR."""
import random
from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from bs4 import BeautifulSoup
from skimage import io


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
        points = element['points'] # type: ignore
        padded_points = pad_points(points, pad_value) # type: ignore
        element['points'] = padded_points # type: ignore

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


def pad_image(image: Image.Image, pad: int = 10) -> Image.Image:
    """
    Pads the given PIL Image with a specified number of pixels on each side.

    Args:
        image (Image.Image): The input PIL Image to be padded.
        pad (int): The number of pixels to pad on each side. Default is 10.

    Returns:
        Image.Image: The padded PIL Image.
    """
    return ImageOps.expand(image, border=(pad, pad, pad, pad), fill=0)


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


def adjust_path(path: str) -> str:
    """
    Make sure, there is no slash at the end of a (folder) spath string.

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


def line_has_text(line: BeautifulSoup) -> bool:
    """
    Checks if line has text in it.

    Args:
        line: BeautifulSoup object representing line

    Returns:
        True if the line has TextEquiv and Unicode xml tags"""
    return bool(line.TextEquiv and line.TextEquiv.Unicode and len(line.TextEquiv.Unicode.contents))


def load_image(image_path: str) -> torch.Tensor:
    """
    Loads an image and ensures it has the right dimensions.

    Args:
        image_path: path to image

    Returns:
        torch tensor of shape (H, W, C) with values in the range [0, 1]
    """
    image = torch.from_numpy(io.imread(image_path))
    # image is black and white
    if image.dim() == 2:
        return image[None, :, :].repeat(3, 1, 1) / 256

    # image has channels last
    if image.shape[-1] == 3:
        return image.permute(2, 0, 1) / 256

    # image has alpha channel and channel last
    if image.shape[-1] == 4:
        return image[:, :, :3].permute(2, 0, 1) / 256

    return image / 256
