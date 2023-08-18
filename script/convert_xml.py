"""
Main Module for converting annotation xml files to numpy images
"""
import argparse
import json
import os
from typing import Dict, List

import numpy as np
from bs4 import BeautifulSoup
from skimage import io  # type: ignore
from tqdm import tqdm  # type: ignore

from script.draw_img import draw_img, LABEL_NAMES  # type: ignore
from script.read_xml import read_transcribus, read_hlna2013  # type: ignore

INPUT = "../Data/input_back/"
OUTPUT = "../Data/Targets_back/"


def main():
    """Load xml files and save result image.
    Calls read and draw functions"""
    read = read_transcribus if args.dataset == 'transcribus' else read_hlna2013
    paths = [f[:-4] for f in os.listdir(INPUT) if f.endswith(".xml")]
    for path in tqdm(paths):
        annotation = read(f'{INPUT}{path}.xml')
        img = draw_img(annotation)
        io.imsave(f'{OUTPUT}{path}.png', img / 10)

        with open(f'{OUTPUT}{path}.json', 'w', encoding="utf-8") as file:
            json.dump(annotation, file)

        # draw image
        img = draw_img(annotation)

        # save image
        np_save(f"{OUTPUT}{path}", img)


def np_save(file: str, img: np.ndarray):
    """
    saves given image in outfile.npy
    :param file: name of the file without ending
    :param img: numpy array to save
    """
    np.save(f"{file}.npy", img)


def img_save(file: str, img: np.ndarray):
    """
    saves given as outfile.png
    :param file: name of the file without ending
    :param img: numpy array to save
    """
    io.imsave(f'{file}.png', img)


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description='creates targets from annotation xmls')
    parser.add_argument('--dataset', '-d', type=str, default='transcribus', help='select dataset to load '
                                                                                 '(transcribus, HLNA2013)')

    return parser.parse_args()


def create_xml(xml_file: str, segmentations: Dict[int, List[List[float]]]) -> BeautifulSoup:
    """
    Creates soup object containing Page Tag and Regions
    :param xml_file: xml file, to which the page data will be written
    :param segmentations: dictionary assigning labels to polygon lists
    :param file_name: image file name
    :param size: image size
    """
    xml_data = BeautifulSoup(xml_file, "xml")
    page = xml_data.find("Page")
    order = xml_data.new_tag("ReadingOrder")
    order_group = xml_data.new_tag("OrderedGroup", attrs={"caption": "Regions reading order"})

    index = 0
    for label, segmentation in segmentations.items():
        for polygon in segmentation:
            order_group.append(xml_data.new_tag("RegionRefIndexed", attrs={"index": str(index)}))
            region = xml_data.new_tag(
                "TextRegion",
                attrs={"custom": f"readingOrder {{index:{index};}} structure {{type:{LABEL_NAMES[label]};}}"})
            region.append(xml_data.new_tag("Coords", attrs={"points": list_to_str_int(polygon)}))
            page.append(region)
            index += 1
    order.append(order_group)
    page.insert(1, order)
    return xml_data


def list_to_str_int(input_list: List[float]) -> str:
    """
    Converts a list to string, while converting each element in the list to an integer.
    :param input_list: list with coordinates
    :return: string
    """
    input_map = map(int, input_list)
    generator_expression = (str(element) for element in input_map)
    string = ", ".join(generator_expression)
    return string


if __name__ == '__main__':
    args = get_args()
    main()
