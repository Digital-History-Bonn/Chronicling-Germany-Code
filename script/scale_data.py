"""Scipt for rescaling images and/or xml data."""
import argparse
import os

import numpy as np
from numpy import ndarray
from PIL import Image
from PIL.Image import BICUBIC # pylint: disable=no-name-in-module
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from src.layout_segmentation.processing.read_xml import xml_polygon_to_polygon_list
from src.layout_segmentation.utils import adjust_path

def rescale(args: argparse.Namespace):
    """Rescale data according to scaling parameter in the provided directory."""
    data_path = adjust_path(args.data_path)
    output_path = adjust_path(args.output_path)
    extension = args.extension

    file_names = [
        f[:-4] for f in os.listdir(data_path) if f.endswith(extension)
    ]
    for name in tqdm(file_names, desc="Rescaling images", unit="image"):
        image = scale_image(args, data_path, extension, name, output_path)
        image.save(output_path + name + extension)


def scale_image(scale: int, data_path: str, extension: str, name: str) -> Image.Image:
    """

    :param scale:
    :param data_path:
    :param extension:
    :param name:
    :param output_path:
    """
    image = Image.open(data_path + name + extension).convert("RGB")
    shape = int(image.size[0] * scale), int(image.size[1] * scale)
    return image.resize(shape, resample=BICUBIC)


def scale_xml(scale: int, data_path: str, name: str, tag_list: List[str]) -> BeautifulSoup:
    """

    :param scale:
    :param data_path:
    :param extension:
    :param name:
    :param output_path:
    """
    with open(f"{data_path + name}.xml", "r", encoding="utf-8") as file:
        data = file.read()

    bs_data = BeautifulSoup(data, "xml")

    for tag in tag_list:
        regions = bs_data.find_all(tag)

        region_text_list = []
        region_coords_list = []
        region_conf_list = []
        lines_data = []

        for region in regions:
            lines = region.find_all("TextLine")
            for line in lines:

def scale_coordinates(tag: Tag, scale: int) -> ndarray:
    polygon = xml_polygon_to_polygon_list(tag)
    polygon_ndarray = np.array(polygon, dtype=int).T




def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="Newspaper Layout Prediction")
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default="data/",
        help="Data path for images.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default="output/",
        help="Output path for images.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".jpg",
        help="Image extension.",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1,
        help="Scaling factor for images.",
    )


if __name__ == "__main__":
    parameter_args = get_args()

    if not os.path.exists(f"{parameter_args.output_path}"):
        os.makedirs(f"{parameter_args.output_path}")

    rescale(parameter_args)