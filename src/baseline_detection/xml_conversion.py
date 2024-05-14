import glob
from pprint import pprint
from typing import List

import numpy as np
from bs4 import BeautifulSoup
from shapely.geometry import Polygon

from src.baseline_detection.mask_rcnn.utils import convert_coord


def get_textregions(page):
    text_regions = page.find_all('TextRegion')
    polygons = [Polygon(convert_coord(elem)) for elem in text_regions]
    return text_regions, polygons


def polygon_to_string(input_list: List[float]) -> str:
    """
    Converts a list to string, while converting each element in the list to an integer. X and y coordinates are
    separated by a comma, each pair is separated from other coordinate pairs by a space. This format is required
    for transkribus
    :param input_list: list withcoordinates
    :return: string
    """
    generator_expression = (
        f"{int(input_list[index])},{int(input_list[index + 1])}"
        for index in range(0, len(input_list), 2)
    )
    string = " ".join(generator_expression)

    return string


def add_baselines(layout_xml, textlines: List[Polygon], baselines: List[np.ndarray]) -> None:
    with open(layout_xml, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')

    # Find and remove all TextLines if exists
    page_elements = page.find_all('TextLine')
    for page_element in page_elements:
        page_element.decompose()

    textregions, polygones = get_textregions(page)

    counter_dict = {i: 0 for i in range(len(textregions))}
    for i, (textline, baseline) in enumerate(zip(textlines, baselines)):
        for j, (textregion, polygon) in enumerate(zip(textregions, polygones)):
            if polygon.contains(textline.centroid):
                counter_dict[j] += 1
                new_textline = soup.new_tag('TextLine')
                new_textline['custom'] = f"readingOrder {{index:{counter_dict[j]};}}"
                new_textline['id'] = textregion['id'] + f'_tl_{counter_dict[j]}'

                # add textline
                coords_element = soup.new_tag("Coords")
                coords_element["points"] = polygon_to_string(np.array(textline.exterior.coords)[:, ::-1].ravel().tolist())
                new_textline.append(coords_element)

                # add baseline
                baseline_element = soup.new_tag("Baseline")
                baseline_element["points"] = polygon_to_string(np.array(baseline.coords)[:, ::-1].ravel().tolist())
                new_textline.append(baseline_element)

                textregion.append(new_textline)
                break

    # Write the modified XML back to file with proper formatting
    with open('../../../data/predictionExample.xml', 'w', encoding='utf-8') as file:
        file.write(soup.prettify())


