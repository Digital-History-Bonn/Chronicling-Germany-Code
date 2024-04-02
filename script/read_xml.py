"""
Module contains read xml functions for all datasets. Data will be writen into a dictionary
mypy typing is ignored for this dictionary
"""
import re
from typing import Dict, List, Tuple, Union

from bs4 import BeautifulSoup, ResultSet


def read_transcribus(
    path: str,
) -> Dict[str, Union[List[int], Dict[str, List[List[List[int]]]]]]:
    """
    reads xml file and returns dictionary containing annotations
    :param path: path to file
    :return: dictionary {height: , width: , tags: {tag_name_1: [], tag_name_2: [], ...}}
    """
    with open(path, "r", encoding="utf-8") as file:
        data = file.read()

    bs_data = BeautifulSoup(data, "xml")
    tags_dict = {"TextLine": []}  # type: ignore

    tags_dict = find_regions(bs_data, "TextRegion", True, "TextLine", tags_dict)
    tags_dict = find_regions(bs_data, "SeparatorRegion", False, "", tags_dict)
    tags_dict = find_regions(bs_data, "Image", False, "", tags_dict)
    tags_dict = find_regions(bs_data, "Graphic", False, "", tags_dict)

    page = bs_data.find("Page")

    if page:
        return {
            "size": [int(page["imageWidth"]), int(page["imageHeight"])],
            "tags": tags_dict,
        }
    return {}


def find_regions(
    data: BeautifulSoup,
    tag: str,
    search_children: bool,
    child_tag: str,
    tags_dict: Dict[str, List[List[List[int]]]],
) -> Dict[str, List[List[List[int]]]]:
    """
    returns dictionary with all coordinates of specified regions
    :param data: BeautifulSoup xml data
    :param tag: tag to be found
    :param search_children: only True if there are children to be included
    :param child_tag: children tag to be found
    :param tags_dict: dictionary to contain region data
    :return: tags: {tag_name_1: [], tag_name_2: [], ...}
    """
    regions = data.find_all(tag)
    for region in regions:
        region_type_matches = re.search(
            r"readingOrder \{index:(.+?);} structure \{type:(.+?);}", region["custom"]
        )
        if region_type_matches is None:
            region_type = "UnknownRegion"
        else:
            region_type = region_type_matches.group(2)
        if region_type not in tags_dict:
            tags_dict[region_type] = []
        tags_dict[region_type].append(
            [pair.split(",") for pair in region.Coords["points"].split()]
        )
        if search_children:
            lines = region.find_all(child_tag)
            if child_tag not in tags_dict:
                tags_dict[child_tag] = []
            for line in lines:
                tags_dict[child_tag].append(
                    [pair.split(",") for pair in line.Coords["points"].split()]
                )
    return tags_dict


def read_hlna2013(
    path: str,
) -> Dict[str, Union[List[int], Dict[str, List[List[Tuple[int, int]]]]]]:
    """
    reads xml file and returns important information in dict
    :param path: path to file
    :return: dict with important information
    """
    annotation: Dict[str, Union[List[int], Dict[str, List[List[Tuple[int, int]]]]]] = {}
    tables: List[List[Tuple[int, int]]] = []
    with open(path, "r", encoding="utf-8") as file:
        data = file.read()

    # read xml
    bs_data = BeautifulSoup(data, "xml")
    annotation["size"] = [
        int(bs_data.find("Page").get("imageHeight")),
        int(bs_data.find("Page").get("imageWidth")),
    ]
    annotation["tags"] = {"table": tables}

    text_regions = bs_data.find_all("TextRegion")
    separator_regions = bs_data.find_all("SeparatorRegion")
    table_regions = bs_data.find_all("TableRegion")

    get_coordinates(annotation, separator_regions, text_regions)

    # get coordinates of all Tables
    for table in table_regions:
        coord = table.find("Coords")
        tables.append(
            [(int(p.get("x")), int(p.get("y"))) for p in coord.find_all("Point")]
        )

    return annotation


def get_coordinates(
    annotation: Dict[str, Union[List[int], Dict[str, List[List[Tuple[int, int]]]]]],
    separator_regions: ResultSet,
    text_regions: ResultSet,
) -> None:
    """Append coordinates to annotation dictionary
    :param annotation: dictionary to contain data
    :param separator_regions: set of coordinates in string format
    :param text_regions: set of coordinates in string format
    """
    # get coordinates of all TextRegions
    paragraphs = []
    headings = []
    header = []
    unknown_region = []
    for sep in text_regions:
        coord = sep.find("Coords")
        if sep.get("type") == "heading":
            headings.append(
                [(int(p.get("x")), int(p.get("y"))) for p in coord.find_all("Point")]
            )
        elif sep.get("type") == "paragraph":
            paragraphs.append(
                [(int(p.get("x")), int(p.get("y"))) for p in coord.find_all("Point")]
            )
        elif sep.get("type") == "header":
            header.append(
                [(int(p.get("x")), int(p.get("y"))) for p in coord.find_all("Point")]
            )
        else:
            unknown_region.append(
                [(int(p.get("x")), int(p.get("y"))) for p in coord.find_all("Point")]
            )
    annotation["tags"]["article"] = paragraphs  # type: ignore
    annotation["tags"]["heading"] = headings  # type: ignore
    annotation["tags"]["header"] = header  # type: ignore
    annotation["tags"]["UnknownRegion"] = unknown_region  # type: ignore
    # get coordinates of all seperators
    separator = []
    for sep in separator_regions:
        coord = sep.find("Coords")
        separator.append(
            [(int(p.get("x")), int(p.get("y"))) for p in coord.find_all("Point")]
        )
    annotation["tags"]["separator_vertical"] = separator  # type: ignore
