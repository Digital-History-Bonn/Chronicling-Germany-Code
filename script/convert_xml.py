"""
Main Module for converting annotation xml files to numpy images. Also contains backwards converting functions, which
take polygon data and convert it to xml.
"""
import argparse
import os
from typing import Dict, List

from skimage.color import label2rgb  # pylint: disable=no-name-in-module
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from skimage import io
from tqdm import tqdm
from numpy import ndarray

from script import draw_img, read_xml
from script.draw_img import LABEL_NAMES

# import draw_img, read_xml
# from draw_img import LABEL_NAMES

cmap = [
    (1.0, 0.0, 0.16),
    (1.0, 0.43843843843843844, 0.0),
    (0, 0.222, 0.222),
    (0.36036036036036045, 0.5, 0.5),
    (0.0, 1.0, 0.2389486260454002),
    (0.8363201911589008, 1.0, 0.0),
    (0.0, 0.5615942028985507, 1.0),
    (0.0422705314009658, 0.0, 1.0),
    (0.6461352657004831, 0.0, 1.0),
    (1.0, 0.0, 0.75),
]

INPUT = "../data/newspaper/annotations/"
OUTPUT = "../data/newspaper/targets/"


def draw_prediction(img: ndarray, path: str) -> None:
    """
    Draw prediction with legend. And save it.
    :param img: prediction ndarray
    :param path: path for the prediction to be saved.
    """

    # unique, counts = np.unique(img, return_counts=True)
    # print(dict(zip(unique, counts)))
    values = LABEL_NAMES
    for i in range(len(values)):
        img[-1][-(i + 1)] = i + 1
    plt.imshow(label2rgb(img, bg_label=0, colors=cmap))
    plt.axis("off")
    # create a patch (proxy artist) for every color
    # patches = [mpatches.Patch(color=cmap[i], label=f"{values[i]}") for i in range(9)]
    # put those patched as legend-handles into the legend
    # plt.legend(handles=patches, bbox_to_anchor=(1.3, -0.10), loc="lower right")
    plt.autoscale(tight=True)
    plt.savefig(path, bbox_inches=0, pad_inches=0, dpi=500)
    # plt.show()


def main(parsed_args: argparse.Namespace) -> None:
    """Load xml files and save result image.
    Calls read and draw functions"""
    read = (
        read_xml.read_transcribus
        if parsed_args.dataset == "transcribus"
        else read_xml.read_hlna2013
    )
    paths = [
        f[:-4] for f in os.listdir(parsed_args.annotations_path) if f.endswith(".xml")
    ]

    if not os.path.exists(parsed_args.output_path):
        print(f"creating {parsed_args.output_path}.")
        os.makedirs(parsed_args.output_path)

    target_paths = [
        f[:-4] for f in os.listdir(parsed_args.output_path) if f.endswith(".npy")
    ]
    for path in tqdm(paths):
        if path in target_paths:
            continue
        annotation: dict = read(f"{parsed_args.annotations_path}{path}.xml")  # type: ignore
        if len(annotation) < 1:
            continue
        img = draw_img.draw_img(annotation)

        # Debug
        draw_prediction(img, f"{parsed_args.output_path}{path}")

        # with open(f"{OUTPUT}{path}.json", "w", encoding="utf-8") as file:
        #     json.dump(annotation, file)

        # save ndarray
        np_save(f"{parsed_args.output_path}{path}", img)


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
    order = xml_data.new_tag("ReadingOrder")
    order_group = xml_data.new_tag(
        "OrderedGroup", attrs={"caption": "Regions reading order"}
    )

    add_regions_to_xml(order_group, page, reading_order, segmentations, xml_data, scale)
    order.append(order_group)
    page.insert(1, order)
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
            region = xml_data.new_tag(
                "TextRegion",
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


if __name__ == "__main__":
    args = get_args()
    main(args)
