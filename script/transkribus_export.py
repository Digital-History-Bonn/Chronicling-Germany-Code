"""Module contains polygon conversion and export functions."""
from typing import Dict, List

import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from numpy import ndarray
from shapely.geometry import Polygon  # type: ignore
from skimage import measure  # type: ignore


def create_sub_masks(mask_image: Image) -> Dict[int, Image]:
    """ Split prediction in to submasks.
    From https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#create-custom-coco-dataset"""
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks: Dict[int, Image] = {}
    for pos_x in range(width):
        for pos_y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((pos_x, pos_y))

            # If the pixel is not black...
            if pixel != 0:
                # Check to see if we've created a sub-mask...
                sub_mask = sub_masks.get(pixel)
                if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contour module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel] = Image.new('1', (width + 2, height + 2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel].putpixel((pos_x + 1, pos_y + 1), 1)

    return sub_masks


def create_polygons(sub_mask: ndarray) -> List[List[float]]:
    """ Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g., an elephant behind a tree)
    # from https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#create-custom-coco-dataset"""
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
    segmentations: List[List[float]] = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in enumerate(contour):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        if len(segmentation) > 2:
            segmentations.append(segmentation)

    return segmentations


def prediction_to_polygons(pred: ndarray) -> Dict[int, List[List[float]]]:
    """
    Converts prediction int ndarray to a dictionary of polygons
    :param pred:
    """
    masks = create_sub_masks(Image.fromarray(pred.astype(np.uint8)))
    segmentations = {}
    for label, mask in masks.items():
        segmentations[label] = create_polygons(np.array(mask))
    return segmentations
