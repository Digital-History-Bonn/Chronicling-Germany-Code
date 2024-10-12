"""Utility functions for baseline detection."""
import random
import re
from typing import Tuple, Union, List, Dict, Optional

import numpy as np
import torch
from bs4 import PageElement, BeautifulSoup
from scipy import ndimage
from shapely import Polygon
from skimage import io

from src.baseline_detection.class_config import TEXT_CLASSES


def order_lines(region: PageElement) -> None:
    """Sort lines by estimating columns and sorting columns from left to right and lines inside a column
    from top to bottom."""
    lines = region.find_all("TextLine")
    if len(lines) == 0:
        return

    properties_list = []
    for i, line in enumerate(lines):
        line_polygon = Polygon([tuple(pair.split(",")) for pair in line.Coords["points"].split()])
        line_centroid = line_polygon.centroid
        bbox = line_polygon.bounds
        properties_list.append(
            [i, bbox[0], bbox[2] - bbox[0], line_centroid.x,
             line_centroid.y])  # index, minx, width, centroidx, centroidy
    properties = np.array(properties_list)
    median_width = np.median(properties[:, 2])
    global_min_x = np.min(properties[:, 1])

    estimated_column_border = global_min_x + median_width
    columns = np.copy(properties)

    result = []
    while True:
        in_column = (columns[:, 1] < (estimated_column_border - median_width / 2)) + (
                    columns[:, 3] < estimated_column_border)

        if all(np.invert(in_column)):
            estimated_column_border += median_width
            continue

        result += columns[in_column][np.argsort(columns[in_column][:, 4])][:, 0].tolist()

        if all(in_column):
            break
        columns = columns[np.invert(in_column)]

    ordered_indices = {int(k): v for v, k in enumerate(result)}

    for i, line in enumerate(lines):
        custom_match = re.search(
            r"(structure \{type:.+?;})", line["custom"]
        )
        class_info = "" if custom_match is None else custom_match.group(1)
        line.attrs['custom'] = f"readingOrder {{index:{ordered_indices[i]};}} {class_info}"


def get_bbox(points: Union[np.ndarray, torch.Tensor],  # type: ignore
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


def is_valid(box: torch.Tensor) -> bool:
    """
    Checks if given bounding box has a valid size.

    Args:
        box: bounding box (xmin, ymin, xmax, ymax)

    Returns:
        True if bounding box is valid
    """
    if box[2] - box[0] <= 0:
        return False
    if box[3] - box[1] <= 0:
        return False
    return True


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


def get_tag(textregion: PageElement) -> str:
    """
    Returns the tag of the given textregion.

    Args:
        textregion: PageElement of Textregion

    Returns:
        Given tag of that Textregion
    """
    desc = textregion['custom']
    match = re.search(r"\{type:.*;\}", desc)
    if match is None:
        return 'UnknownRegion'
    return match.group()[6:-2]


def get_reading_order_idx(textregion: PageElement) -> int:
    """
    Extracts reading order from textregion PageElement.

    Args:
        textregion: PageElement of Textregion

    Returns:
         Reading Order Index as int
    """
    desc = textregion['custom']
    match = re.search(r"readingOrder\s*\{index:(\d+);\}", desc)
    if match is None:
        return -1
    return int(match.group(1))


def extract(xml_path: str
            ) -> Tuple[List[Dict[str, Union[torch.Tensor, List[torch.Tensor], int]]],
List[torch.Tensor]]:
    """
    Extracts the annotation from the xml file.

    Args:
        xml_path: path to the xml file.

    Returns:
        A list of dictionary representing all Textregions in the given document
        A list of polygons as torch tensors for masking areas
    """
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    paragraphs = []
    mask_regions = []

    text_regions = page.find_all(['TextRegion', 'TableRegion'])
    for region in text_regions:
        tag = get_tag(region)
        coords = region.find('Coords')
        region_polygon = torch.tensor([tuple(map(int, point.split(','))) for
                                       point in coords['points'].split()])[:, torch.tensor([1, 0])]

        if tag in ['table']:
            if is_valid(torch.tensor(get_bbox(region_polygon))):
                mask_regions.append(region_polygon)

        if tag in TEXT_CLASSES:
            region_bbox = torch.tensor(get_bbox(region_polygon))

            if is_valid(region_bbox):
                region_dict = extract_region(region, region_bbox)
                region_dict['textregion'] = region_polygon

                # only adding regions with at least one textline
                if len(region_dict['textline_polygone']) > 0:  # type: ignore
                    paragraphs.append(region_dict)

    return paragraphs, mask_regions


def extract_region(region: BeautifulSoup, region_bbox: torch.Tensor) -> Dict[
    str, Union[torch.Tensor, List[torch.Tensor], int]]:
    """
    Extracts the annotation data for a given region.

    Args:
        region: BeautifulSoup object representing a Textregion
        region_bbox: torch tensor of shape (N, 2) containing a list of coordinates

    Returns:
        Region dict with annotation data of region.
    """
    region_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor], int]] = {
        'region_bbox': region_bbox,
        'bboxes': [],
        'textline_polygone': [],
        'baselines': [],
        'readingOrder': get_reading_order_idx(region)}
    text_region = region.find_all('TextLine')
    for text_line in text_region:
        polygon = text_line.find('Coords')
        baseline = text_line.find('Baseline')
        if baseline:
            # get and shift baseline
            line = torch.tensor([tuple(map(int, point.split(','))) for
                                 point in baseline['points'].split()])
            line = line[:, torch.tensor([1, 0])]

            region_dict['baselines'].append(line)  # type: ignore

            # get mask
            polygon_pt = torch.tensor([tuple(map(int, point.split(','))) for
                                       point in polygon['points'].split()])
            polygon_pt = polygon_pt[:, torch.tensor([1, 0])]

            # calc bbox for line
            box = torch.tensor(get_bbox(polygon_pt))[torch.tensor([1, 0, 3, 2])]
            box = box.clip(min=0)

            # add bbox to data
            if is_valid(box):
                region_dict['bboxes'].append(box)  # type: ignore

                # add mask to data
                region_dict['textline_polygone'].append(polygon_pt)  # type: ignore
    return region_dict


def nonmaxima_suppression(input_array: np.ndarray,
                          element_size: Tuple[int, int] = (7, 1)) -> np.ndarray:
    """
    From https://github.com/DCGM/pero-ocr/blob/master/pero_ocr/layout_engines/cnn_layout_engine.py.

    Vertical non-maxima suppression.

    Args:
        input_array: input array
        element_size: structure element for greyscale dilations

    Returns:
        non maxima suppression of baseline input image
    """
    if len(input_array.shape) == 3:
        dilated = np.zeros_like(input_array)
        for i in range(input_array.shape[0]):
            dilated[i, :, :] = ndimage.grey_dilation(
                input_array[i, :, :], size=element_size)
    else:
        dilated = ndimage.grey_dilation(input_array, size=element_size)

    return input_array * (input_array == dilated)  # type: ignore


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
