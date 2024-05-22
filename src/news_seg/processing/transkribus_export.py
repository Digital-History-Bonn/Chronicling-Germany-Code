"""Module for creating Transkribus PAGE XML data."""
import argparse
import os
import re
from typing import Dict, List

import numpy as np
from bs4 import BeautifulSoup
from shapely import Polygon, centroid

from src.news_seg.class_config import LABEL_NAMES, REGION_TYPES


def export_xml(args: argparse.Namespace, file: str, reading_order_dict: Dict[int, int],
               segmentations: Dict[int, List[List[float]]]) -> None:
    """
    Open pre created transkribus xml files and save polygon xml data.
    :param args: args
    :param file: xml path
    :param reading_order_dict: reading order value for each index
    :param segmentations: polygon dictionary sorted by labels
    """
    with open(
            f"{args.data_path}page/{os.path.splitext(file)[0]}.xml",
            "r",
            encoding="utf-8",
    ) as xml_file:
        xml_data = create_xml(xml_file.read(), segmentations, reading_order_dict, args.scale)
    with open(
            f"{args.data_path}page/{os.path.splitext(file)[0]}.xml",
            "w",
            encoding="utf-8",
    ) as xml_file:
        xml_file.write(xml_data.prettify())


def create_xml(
        xml_file: str, segmentations: Dict[int, List[List[float]]], reading_order: Dict[int, int], scale: float
) -> BeautifulSoup:
    """
    Creates a soup object containing Page Tag and Regions
    :param xml_file: xml file, to which the page data will be written
    :param segmentations: dictionary assigning labels to polygon lists
    :param file_name: image file name
    :param size: image size
    """
    xml_data = BeautifulSoup(xml_file, "xml")
    page = xml_data.find("Page")
    page.clear()
    order = xml_data.new_tag("ReadingOrder")
    order_group = xml_data.new_tag(
        "OrderedGroup", attrs={"caption": "Regions reading order"}
    )

    add_regions_to_xml(order_group, page, reading_order, segmentations, xml_data, scale)
    order.append(order_group)
    page.insert(0, order)
    return xml_data


def add_regions_to_xml(order_group: BeautifulSoup, page: BeautifulSoup, reading_order: Dict[int, int],
                       segmentations: Dict[int, List[List[float]]], xml_data: BeautifulSoup, scale: float) -> None:
    """
    Add ReadingOrder XML and Text Region List to Page
    :param order_group: BeautifulSOup Object for ReadingOrder
    :param page: Page BeautifulSOup Object
    :param reading_order: dict
    :param segmentations: dictionary assigning labels to polygon lists
    :param xml_data: final BeautifulSOup object
    """
    index = 0
    for label, segmentation in segmentations.items():
        for polygon in segmentation:
            order_group.append(
                xml_data.new_tag(
                    "RegionRefIndexed",
                    attrs={"index": str(reading_order[index]), "regionRef": str(index)},
                )
            )

            region_type = REGION_TYPES[get_label_name(label)]

            region = xml_data.new_tag(
                region_type,
                attrs={
                    "id": str(index),
                    "custom": f"readingOrder {{index:{reading_order[index]};}} structure "
                              f"{{type:{get_label_name(label)};}}",
                },
            )
            region.append(
                xml_data.new_tag("Coords", attrs={"points": polygon_to_string(polygon, scale)})
            )
            page.append(region)
            index += 1


def get_label_name(label: int) -> str:
    """
    Get label name from LABEL_NAMES list
    :param label: int label value
    :return: label name
    """
    return LABEL_NAMES[label - 1]


def polygon_to_string(input_list: List[float], scale: float) -> str:
    """
    Converts a list to string, while converting each element in the list to an integer. X and y coordinates are
    separated by a comma, each pair is separated from other coordinate pairs by a space. This format is required
    for transkribus
    :param input_list: list withcoordinates
    :return: string
    """
    generator_expression = (
        f"{int(input_list[index] * scale**-1)},{int(input_list[index + 1] * scale**-1)}"
        for index in range(0, len(input_list), 2)
    )
    string = " ".join(generator_expression)

    return string


def copy_xml(bs_copy: BeautifulSoup, bs_data: BeautifulSoup, id_list: List[str],
             reading_order_dict: dict[int, int]) -> None:
    """
    Copy regions into new BeautifulSoup object with corrected reading order.
    :param bs_copy: copy of xml data, to be overwritten
    :param bs_data: xml data
    :param id_list: list of region ids, that are used to copy data.
    :param reading_order_dict:
    """
    page = bs_copy.find("Page")
    page.clear()
    order_group = bs_copy.new_tag(
        "OrderedGroup", attrs={"caption": "Regions reading order"}
    )
    for key, order in reading_order_dict.items():
        region = bs_data.find(attrs={'id': f'{id_list[int(key)]}'})

        sort_lines(region)

        custom_match = re.search(
            r"(structure \{type:.+?;})", region["custom"]
        )

        class_info = "structure {type:UnkownRegion;}" if custom_match is None else custom_match.group(1)
        region.attrs['custom'] = f"readingOrder {{index:{order};}} {class_info}"

        order_group.append(
            bs_copy.new_tag(
                "RegionRefIndexed",
                attrs={"index": str(order), "regionRef": id_list[int(key)]},
            )
        )
        page.append(region)


def sort_lines(region: BeautifulSoup) -> None:
    """Sort lines by ascending height."""
    lines = region.find_all("TextLine")
    height_list = []
    for line in lines:
        line_polygon = Polygon([tuple(pair.split(",")) for pair in line.Coords["points"].split()])
        height_list.append(centroid(line_polygon).y)
    sorted_heights = {int(k): v for v, k in enumerate(np.argsort(np.array(height_list, dtype=int)))}
    for i, line in enumerate(lines):
        custom_match = re.search(
            r"(structure \{type:.+?;})", line["custom"]
        )
        class_info = "" if custom_match is None else custom_match.group(1)
        line.attrs['custom'] = f"readingOrder {{index:{sorted_heights[i]};}} {class_info}"
