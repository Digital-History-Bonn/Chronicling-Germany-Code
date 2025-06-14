"""Utility functions for baseline detection."""
import random
import re
from typing import Tuple, Union, List, Dict, Optional

import bs4
import numpy as np
import torch
from bs4 import PageElement, BeautifulSoup
from scipy import ndimage
from shapely.geometry import Polygon, LineString
from skimage import io

from src.cgprocess.baseline_detection.class_config import TEXT_CLASSES
from src.cgprocess.layout_segmentation.processing.read_xml import xml_polygon_to_polygon_list


def order_lines(region: bs4.element.Tag) -> None:
    """Sort lines by estimating columns and sorting columns from left to right and lines inside a column
    from top to bottom."""
    lines = region.find_all("TextLine")

    properties_list = []
    for i, line in enumerate(lines):
        line_polygon = Polygon(xml_polygon_to_polygon_list(line.Coords["points"]))
        line_centroid = line_polygon.centroid
        bbox = line_polygon.bounds
        # pylint: disable=no-member
        properties_list.append(
            [i, bbox[0], bbox[2] - bbox[0], line_centroid.x,
             line_centroid.y])  # index, minx, width, centroidx, centroidy
    if len(properties_list) == 0:
        return
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
            r"(structure \{type:.+?;})", line["custom"] # type: ignore
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



def get_tag(textregion: PageElement) -> str:
    """
    Returns the tag of the given textregion.

    Args:
        textregion: PageElement of Textregion

    Returns:
        Given tag of that Textregion
    """
    desc = textregion['custom'] # type: ignore
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
    desc = textregion['custom'] # type: ignore
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

    text_regions = page.find_all(['TextRegion', 'TableRegion']) # type: ignore
    for region in text_regions:
        tag = get_tag(region)
        region_polygon = torch.tensor(xml_polygon_to_polygon_list(region.Coords["points"]))[:, torch.tensor([1, 0])]

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
        polygon = text_line.find('Coords') # type: ignore
        baseline = text_line.find('Baseline') # type: ignore
        if baseline:
            # get and shift baseline
            line = torch.tensor(xml_polygon_to_polygon_list(baseline["points"])) # type: ignore
            line = line[:, torch.tensor([1, 0])]

            region_dict['baselines'].append(line)  # type: ignore

            # get mask
            polygon_pt = torch.tensor(xml_polygon_to_polygon_list(polygon["points"])) # type: ignore
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


def polygon_to_string(input_list: List[float]) -> str:
    """
    Converts a polygon into a string to save in the xml file.

    Converts a list to string, while converting each element in the list to an integer.
    X and y coordinates are separated by a comma, each pair is separated from other
    coordinate pairs by a space. This format is required for transkribus.

    Args:
        input_list: list withcoordinates

    Returns:
         Coordinates as a string
    """
    generator_expression = (
        f"{int(input_list[index])},{int(input_list[index + 1])}"
        for index in range(0, len(input_list), 2)
    )
    string = " ".join(generator_expression)

    return string


def add_baselines(layout_xml: str,
                  output_file: str,
                  textlines: List[List[Polygon]], baselines: List[List[LineString]]) -> None:
    """
    Adds testline and baseline prediction form model to the layout xml file.

    Args:
        layout_xml: path to layout xml file
        output_file: path to output xml file
        textlines: list of list of shapely LineString predicted by model
        baselines: list of list of shapely baselines predicted by model
    """
    with open(layout_xml, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')

    # Find and remove all TextLines if exists
    page_elements = page.find_all('TextLine') # type: ignore
    for page_element in page_elements:
        page_element.decompose()

    textregions = page.find_all('TextRegion') # type: ignore

    # adds all predicted textlines to annotation if they have their center inside a text region
    for textregion, region_textlines, region_baselines in zip(textregions, textlines, baselines):
        for i, (textline, baseline) in enumerate(zip(region_textlines, region_baselines)):
            new_textline = soup.new_tag('TextLine')
            new_textline['custom'] = f"readingOrder {{index:{i};}}"
            new_textline['id'] = textregion['id'] + f'_tl_{i}'

            # add textline
            coords_element = soup.new_tag("Coords")
            points_list = np.array(textline.exterior.coords).ravel().tolist()
            coords_element["points"] = polygon_to_string(points_list)
            new_textline.append(coords_element)

            # add baseline
            baseline_element = soup.new_tag("Baseline")
            points_list = np.array(baseline.coords).ravel().tolist()
            baseline_element["points"] = polygon_to_string(points_list)
            new_textline.append(baseline_element)

            textregion.append(new_textline)

    for region in textregions:
        order_lines(region)

    # Write the modified XML back to file with proper formatting
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(soup.prettify()) # type: ignore


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
