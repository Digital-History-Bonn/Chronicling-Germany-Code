"""Script to convert baseline predictions into xml format."""

from typing import List

import numpy as np
from bs4 import BeautifulSoup
from shapely import LineString
from shapely.geometry import Polygon

from src.baseline_detection.utils import order_lines


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
    page_elements = page.find_all('TextLine')
    for page_element in page_elements:
        page_element.decompose()

    textregions = page.find_all('TextRegion')

    # adds all predicted textlines to annotation if they have their center inside a text region
    for textregion, region_textlines, region_baselines in zip(textregions, textlines, baselines):
        for i, (textline, baseline) in enumerate(zip(region_textlines, region_baselines)):
            new_textline = soup.new_tag('TextLine')
            new_textline['custom'] = f"readingOrder {{index:{i};}}"
            new_textline['id'] = textregion['id'] + f'_tl_{i}'

            # add textline
            coords_element = soup.new_tag("Coords")
            points_list = np.array(textline.exterior.coords)[:, ::-1].ravel().tolist()
            coords_element["points"] = polygon_to_string(points_list)
            new_textline.append(coords_element)

            # add baseline
            baseline_element = soup.new_tag("Baseline")
            points_list = np.array(baseline.coords)[:, ::-1].ravel().tolist()
            baseline_element["points"] = polygon_to_string(points_list)
            new_textline.append(baseline_element)

            textregion.append(new_textline)

    for region in textregions:
        order_lines(region)

    # Write the modified XML back to file with proper formatting
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(soup.prettify())
