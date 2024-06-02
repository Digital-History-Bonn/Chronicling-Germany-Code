"""Module for reading xml Files."""
# pylint: disable=duplicate-code
from typing import List, Tuple

from bs4 import BeautifulSoup


def read_lines(
        path: str,
        tag: str,
        child_tag: str,
        confidence: bool = False
) -> Tuple[List[List[str]], List[List[List[Tuple]]], List[List[float]], List[BeautifulSoup]]:
    """
    Reads xml file and extracts all the lines from specified regions.
    :param confidence_threshold: lines below this threshold are ignored
    :param path: path to xml file without xml ending
    :param tag: tag to be found
    :param child_tag: children tag to be found
    :return: Region lists, containing a List of text, line coordinates and confidence for each region seperatly, as
    well as the complete beautiful soup object for all lines
    """
    with open(f"{path}.xml", "r", encoding="utf-8") as file:
        data = file.read()

    bs_data = BeautifulSoup(data, "xml")

    regions = bs_data.find_all(tag)

    region_text_list = []
    region_coords_list = []
    region_conf_list = []
    lines_data = []

    for region in regions:
        lines = region.find_all(child_tag)
        text_list = []
        coords_list = []
        conf_list = []
        for line in lines:
            if line_has_text(line):
                text_list.append(line.TextEquiv.Unicode.contents[0])
                coords_list.append([tuple(pair.split(",")) for pair in line.Coords["points"].split()])
                conf_list.append(line_confidence(confidence, line))
                lines_data.append(line)
        region_text_list.append(text_list)
        region_coords_list.append(coords_list)
        region_conf_list.append(conf_list)
    return region_text_list, region_coords_list, region_conf_list, lines_data


def line_confidence(confidence, line) -> float:
    """Returns True if threshold ist not supplied or no confidence for the line is known. If both are present it only
    returns true, if the confidence is grater or equal to the threshold."""
    if not confidence:
        return 0.0
    if "conf" in line.TextEquiv.attrs:
        return float(line.TextEquiv.attrs["conf"])
    return 0.0


def line_has_text(line):
    """return True if the line has TextEquiv and Unicode xml tags"""
    return bool(line.TextEquiv and line.TextEquiv.Unicode and len(line.TextEquiv.Unicode.contents))
