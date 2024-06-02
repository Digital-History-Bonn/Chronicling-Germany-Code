"""Aplies changes to xml regions, like setting uncertain from comparison of 2 xml files, and renaming region
classes."""

import argparse
import csv
import json
import os
import re
import warnings
from typing import Tuple, List

import numpy as np
from numpy import ndarray
from bs4 import BeautifulSoup, ResultSet
from tqdm import tqdm

from src.OCR.evaluate.read_xml import line_has_text
from src.OCR.utils import adjust_path


def extract_text(xml_data: BeautifulSoup, path: str) -> Tuple[ndarray, list]:
    """
    Sorts regions and lines by the supplied reading order and assembles ndarray for csv export and
    dictionary for json export.
    """
    text_regions = xml_data.find_all("TextRegion")

    region_order = sort_xml_elements(text_regions)

    region_csv: list = []
    region_list: list = []
    for i, region_id in enumerate(region_order):
        region = xml_data.find(attrs={'id': f'{region_id}'})
        lines = region.find_all("TextLine")
        if len(lines) == 0:
            continue
        lines_order = sort_xml_elements(lines)

        region_text_list = []
        region_line_coords = []
        for _, line_id in enumerate(lines_order):
            line = xml_data.find(attrs={'id': f'{line_id}'})
            if line_has_text(line):
                region_text_list.append(line.TextEquiv.Unicode.contents[0])
                region_line_coords.append(line.Coords["points"])

        region_type = re.search(
            r"structure \{type:(.+?);}", region["custom"]
        )
        region_class = "UnkownRegion" if not region_type else region_type.group(1)

        prepare_csv_data(i, path, region_class, region_csv, region_text_list, region_line_coords)

        region_list.append({"class": region_class, "lines": region_text_list})

    return np.vstack(region_csv), region_list


def prepare_csv_data(i: int, path: str, region_class: str, region_csv: list, region_text_list: List[str],
                     region_line_coords: List[str]) -> None:
    """
    Assemble csv data for a single region. This means adding the current region id to all lines of that
    region, as well ass the class and page path.
    :param region_csv: csv array for this page, that contains ndarrays for every region.
    """
    path_array = np.full(len(region_text_list), path)
    region_id_array = np.full(len(region_text_list), str(i))
    class_array = np.full(len(region_text_list), region_class)
    line_id_array = np.char.mod('%d', (np.arange(len(region_text_list)) + 1))
    region_csv.append(
        np.vstack([path_array, region_id_array, line_id_array, np.array(region_text_list), class_array,
                   np.array(region_line_coords)]).T)


def sort_xml_elements(elements: ResultSet) -> ndarray:
    """
    Sort xml elements by readingOrder.
    :param elements: regions or lines list.
    """
    reading_list = []
    for element in elements:

        reading = re.search(
            r"readingOrder \{index:(.+?);}", element["custom"]
        )

        if reading:
            reading_list.append([int(reading.group(1)), element["id"]])
        else:
            warnings.warn("No reading Order found. This line will be ignored.")
    order = np.array(reading_list)
    return order[:, 1][np.argsort(order[:, 0])]  #type: ignore


def main(args: argparse.Namespace) -> None:
    """
    Load xml files and assemble page lists before saving them.
    """
    if not os.path.exists(args.output_path):
        print(f"creating {args.output_path}.")
        os.makedirs(args.output_path)

    paths = [
        f[:-4] for f in os.listdir(args.data_path) if f.endswith(".xml")
    ]

    page_csv = []
    page_json = []

    data_path = adjust_path(args.data_path)
    output_path = adjust_path(args.output_path)

    for path in tqdm(paths):
        with open(f"{data_path}{path}.xml", "r", encoding="utf-8") as file:
            data = file.read()

        xml_data = BeautifulSoup(data, "xml")
        region_csv, region_json = extract_text(xml_data, path)
        page_csv.append(region_csv)
        page_json.append({"path": path, "regions": region_json})

    csv_data = np.vstack(page_csv).tolist()

    with open(f"{output_path}text_export.csv", 'w', newline='', encoding='utf-8') as file:
        data_writer = csv.writer(file, delimiter=';')
        data_writer.writerow(["path", "region", "line", "text", "class"])
        for row in csv_data:
            data_writer.writerow(row)

    with open(f"{output_path}text_export.json", 'w', newline='', encoding='utf-8') as file:
        json.dump(page_json, file)


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default="data/lines/",
        help="path for input xml folder",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        dest="output_path",
        default="data/output-text/",
        help="path for output folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parameter_args = get_args()
    main(parameter_args)
