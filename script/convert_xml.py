"""
Main Module for converting annotation xml files to numpy images. Also contains backwards converting functions, which
take polygon data and convert it to xml.
"""
import argparse
import os
from typing import Dict, List

import numpy as np
from bs4 import BeautifulSoup
from skimage import io
from tqdm import tqdm


# import draw_img, read_xml
# from draw_img import LABEL_NAMES

from script import draw_img, read_xml
from script.draw_img import LABEL_NAMES

INPUT = "../../data/newspaper/annotations/"
OUTPUT = "../../data/newspaper/targets/"


def main(args) -> None:
    """Load xml files and save result image.
    Calls read and draw functions"""
    read = (
        read_xml.read_transcribus
        if args.dataset == "transcribus"
        else read_xml.read_hlna2013
    )
    paths = [f[:-4] for f in os.listdir(args.annotations_path) if f.endswith(".xml")]

    if not os.path.exists(args.output_path):
        print(f'creating {args.output_path}.')
        os.makedirs(args.output_path)

    target_paths = [f[:-4] for f in os.listdir(args.output_path) if f.endswith(".npy")]
    for path in tqdm(paths):
        if path in target_paths:
            continue
        annotation: dict = read(f"{args.annotations_path}{path}.xml")  # type: ignore
        if len(annotation) < 1:
            continue
        img = draw_img.draw_img(annotation)

        # Debug
        # io.imsave(f"{OUTPUT}{path}.png", img / 10)
        #
        # with open(f"{OUTPUT}{path}.json", "w", encoding="utf-8") as file:
        #     json.dump(annotation, file)


        # save ndarray
        np_save(f"{args.output_path}{path}", img)


def np_save(file: str, img: np.ndarray) -> None:
    """
    saves given image in outfile.npy
    :param file: name of the file without ending
    :param img: numpy array to save
    """
    np.save(f"{file}.npy", img)


def img_save(file: str, img: np.ndarray) -> None:
    """
    saves given as outfile.png
    :param file: name of the file without ending
    :param img: numpy array to save
    """
    io.imsave(f"{file}.png", img)


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="creates targets from annotation xmls")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="transcribus",
        help="select dataset to load " "(transcribus, HLNA2013)",
    )
    parser.add_argument(
        "--annotations-path",
        "-a",
        type=str,
        dest="annotations_path",
        default=INPUT,
        help="path for folder with annotations",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        dest="output_path",
        default=OUTPUT,
        help="path for ouput folder",
    )
    return parser.parse_args()


def create_xml(
    xml_file: str, segmentations: Dict[int, List[List[float]]]
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
    order = xml_data.new_tag("ReadingOrder")
    order_group = xml_data.new_tag(
        "OrderedGroup", attrs={"caption": "Regions reading order"}
    )

    index = 0
    for label, segmentation in segmentations.items():
        for polygon in segmentation:
            order_group.append(
                xml_data.new_tag(
                    "RegionRefIndexed",
                    attrs={"index": str(index), "regionRef": str(index)},
                )
            )
            region = xml_data.new_tag(
                "TextRegion",
                attrs={
                    "id": str(index),
                    "custom": f"readingOrder {{index:{index};}} structure {{type:{LABEL_NAMES[label]};}}",
                },
            )
            region.append(
                xml_data.new_tag("Coords", attrs={"points": polygon_to_string(polygon)})
            )
            page.append(region)
            index += 1
    order.append(order_group)
    page.insert(1, order)
    return xml_data


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


if __name__ == "__main__":
    args = get_args()
    main(args)
