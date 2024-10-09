"""
Module contains read xml functions for all datasets. Data will be writen into a dictionary
mypy typing is ignored for this dictionary
"""
import re
from typing import Dict, List, Tuple, Union

from bs4 import BeautifulSoup, ResultSet, Tag
from shapely import Polygon

from src.layout_segmentation.class_config import VALID_TAGS, LABEL_ASSIGNMENTS


def read_transkribus(
        path: str,
        log: bool,
) -> Dict[str, Union[List[int], Dict[str, List[List[List[str]]]]]]:
    """
    reads xml file and returns dictionary containing annotations
    :param path: path to file
    :param log: activates tag check logging
    :return: dictionary {height: , width: , tags: {tag_name_1: [], tag_name_2: [], ...}}
    """
    with open(path, "r", encoding="utf-8") as file:
        data = file.read()

    bs_data = BeautifulSoup(data, "xml")
    tags_dict = {"TextLine": []}  # type: ignore

    tags_dict = find_regions(bs_data, "TextRegion", True, "TextLine", tags_dict)
    tags_dict = find_regions(bs_data, "SeparatorRegion", False, "", tags_dict)
    tags_dict = find_regions(bs_data, "ImageRegion", False, "", tags_dict)
    tags_dict = find_regions(bs_data, "GraphicRegion", False, "", tags_dict)
    tags_dict = find_regions(bs_data, "TableRegion", False, "", tags_dict)

    page = bs_data.find("Page")

    if log:
        check_tags(page, tags_dict)

    if page:
        return {
            "size": [int(page["imageWidth"]), int(page["imageHeight"])],
            "tags": tags_dict,
        }
    return {}


def check_tags(page: BeautifulSoup, tags_dict: Dict[str, List[List[List[str]]]]) -> None:
    """
    Logs all occurrences of unknown regions.
    """
    if "UnknownRegion" in tags_dict.keys():
        print(f"Found {len(tags_dict['UnknownRegion'])} UnknownRegion(s) in {page['imageFilename']}.")
    unknown_tags = [key for key in tags_dict.keys() if key not in VALID_TAGS]
    for tag in unknown_tags:
        print(f'Found {len(tags_dict[tag])} region(s) with unknown {tag} tag in {page["imageFilename"]}.')


def find_regions(
        data: BeautifulSoup,
        tag: str,
        search_children: bool,
        child_tag: str,
        tags_dict: Dict[str, List[List[List[str]]]],
        id_dict: Union[None, Dict[str, List[str]]] = None
) -> Dict[str, List[List[List[str]]]]:
    """
    returns dictionary with all coordinates of specified regions
    :param data: BeautifulSoup xml data
    :param tag: tag to be found
    :param search_children: only True if there are children to be included
    :param child_tag: children tag to be found
    :param tags_dict: dictionary to contain region data
    :param id_dict: if a dict is provided, it will be filled with region ids.
    :return: tags: {tag_name_1: [], tag_name_2: [], ...}
    """
    assert id_dict is None or not child_tag, ("Child tag extraction and id_dict are not compatible. Please be sure "
                                              "you know what you are doing, bevore activating both.")
    regions = data.find_all(tag)

    for region in regions:
        region_type_matches = re.search(
            r"readingOrder \{index:(.+?);} structure \{type:(.+?);}", region["custom"]
        )
        if region_type_matches is None:
            region_type = 'image' if tag in ('ImageRegion', 'GraphicRegion') else "UnknownRegion"
        else:
            region_type = region_type_matches.group(2)
        if region_type not in tags_dict:
            tags_dict[region_type] = []
        tags_dict[region_type].append(
            xml_polygon_to_polygon_list(region)
        )
        if id_dict is not None:
            if region_type not in id_dict:
                id_dict[region_type] = []
            id_dict[region_type].append(
                region["id"]
            )

        if search_children:
            lines = region.find_all(child_tag)
            if child_tag not in tags_dict:
                tags_dict[child_tag] = []
            for line in lines:
                tags_dict[child_tag].append(
                    xml_polygon_to_polygon_list(line)
                )
    return tags_dict


def xml_polygon_to_polygon_list(xml_data: Tag) -> List[List[str]]:
    """
    Splits xml polygon coordinate string to create a polygon, this being a list of coordinate pairs.
    """
    return [pair.split(",") for pair in xml_data.Coords["points"].split()]


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


def read_regions_for_reading_order(
        path: str,
) -> Tuple[Dict[int, List[List[float]]], Dict[int, List[str]], BeautifulSoup]:
    """
    Returns bbox and id dictionaries, that are suited for reading order module. Returns bs_data as well, to preserve
    the original data and only adjust the reading order.
    :param path: path to xml file without .xml
    """
    with open(f"{path}.xml", "r", encoding="utf-8") as file:
        data = file.read()

    tags_dict = {"TextLine": []}  # type: ignore
    id_dict: Dict[str, List[str]] = {}
    bs_data = BeautifulSoup(data, "xml")

    tags_dict = find_regions(bs_data, "TextRegion", False, "", tags_dict, id_dict)
    tags_dict = find_regions(bs_data, "SeparatorRegion", False, "", tags_dict, id_dict)
    tags_dict = find_regions(bs_data, "ImageRegion", False, "", tags_dict, id_dict)
    tags_dict = find_regions(bs_data, "GraphicRegion", False, "", tags_dict, id_dict)
    tags_dict = find_regions(bs_data, "TableRegion", False, "", tags_dict, id_dict)

    bbox_dict, id_align_dict = tag_to_label_dict(id_dict, tags_dict)

    return bbox_dict, id_align_dict, bs_data


def read_raw_data(path: str) -> BeautifulSoup:
    """Read xml file and return BeautifulSoup object"""
    with open(f"{path}.xml", "r", encoding="utf-8") as file:
        data = file.read()

    return BeautifulSoup(data, "xml")


def tag_to_label_dict(id_dict: Dict[str, List[str]], tags_dict: Dict[str, List[List[List[int]]]]) -> Tuple[
    Dict[int, List[List[float]]], Dict[int, List[str]]]:
    """
    Assigns numerical labels to string tags and adds them to dictionaries. Unkown tags are interpreted as
    UnkownRegion (label = 1.)
    :param id_dict:
    :param tags_dict:
    """
    bbox_dict: Dict[int, List[List[float]]] = {1: []}
    id_align_dict: Dict[int, List[str]] = {1: []}

    # UnkownRegion
    for key, coords in tags_dict.items():
        if key not in LABEL_ASSIGNMENTS:
            for coord, region_id in zip(coords, id_dict[key]):
                if len(coord) > 2:
                    bbox_dict[1].append(Polygon(coord).bounds)
                    id_align_dict[1].append(region_id)

    # Valid tags
    for key, label in LABEL_ASSIGNMENTS.items():
        if label != 0 and key in tags_dict:
            if not label in bbox_dict:
                bbox_dict[label] = []
                id_align_dict[label] = []
            for coord, region_id in zip(tags_dict[key], id_dict[key]):
                if len(coord) > 2:
                    bbox_dict[label].append(Polygon(coord).bounds)
                    id_align_dict[label].append(region_id)

    if len(bbox_dict) == 1 and len(bbox_dict[1]) == 0:
        return {}, {}
    return bbox_dict, id_align_dict
