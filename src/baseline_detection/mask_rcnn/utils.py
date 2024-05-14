"""Utility functions for Mask R-CNN baselines detection."""

from typing import Dict, Tuple, Union

import numpy as np
import torch
from bs4 import PageElement
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes


def draw_prediction(image: torch.Tensor, prediction: Dict[str, torch.Tensor]):
    """
    Draws a visualisation of the prediction.

    Args:
        image: image
        prediction: Dict with the prediction of the model

    Raises:
        ValueError: If image is not a color image with 3 chanels

    Returns:
        visualisation of the prediction
    """
    image = image.clone()

    # unbatch image if image has batch dim
    image = image[0] if image.dim() == 4 else image

    # move color channel first if color channel is last
    image = image.permute(2, 0, 1) if image.shape[2] == 3 else image

    # if first dim doesn't have 3 raise Error
    if image.shape[0] != 3:
        raise ValueError("Only RGB image, need to have 3 channels in dim 0 or 2")

    # map [0, 1] to [0, 255]
    if image.max() <= 1.0:
        image *= 256

    if 'masks' in prediction.keys():
        image = draw_segmentation_masks(image.to(torch.uint8), prediction['masks'].squeeze() > .5)

    image = draw_bounding_boxes(image.to(torch.uint8), prediction['boxes'], width=2, colors='red')

    return image


def get_bbox(
        points: Union[np.ndarray, torch.Tensor],  # type: ignore
) -> Tuple[int, int, int, int]:
    """
    Creates a bounding box around all given points.

    Args:
        points: np.ndarray of shape (N x 2) containing a list of points

    Returns:
        coordinates of bounding box in the format (x_min, y_min, x_max, y_max)

    """
    x_max, x_min = points[:, 0].max(), points[:, 0].min()
    y_max, y_min = points[:, 1].max(), points[:, 1].min()

    return x_min, y_min, x_max, y_max  # type: ignore


def convert_coord(element: PageElement) -> np.ndarray:
    """
    Converts PageElement with Coords in to a numpy array.

    Args:
        element: PageElement with Coords for example a Textregion

    Returns:
        np.ndarray of shape (N x 2) containing a list of coordinates
    """
    coords = element.find('Coords')
    return np.array([tuple(map(int, point.split(','))) for
                     point in coords['points'].split()])[:, np.array([1, 0])]
