"""Scipt for rescaling images and/or xml data."""

import argparse
import os
from typing import List

import numpy as np
from bs4 import BeautifulSoup, Tag
from PIL import Image
from PIL.Image import BICUBIC  # pylint: disable=no-name-in-module
from tqdm import tqdm

from cgprocess.layout_segmentation.convert_xml import save_xml
from cgprocess.layout_segmentation.processing.transkribus_export import (
    polygon_to_string,
)
from cgprocess.layout_segmentation.utils import adjust_path
from src.cgprocess.shared.utils import xml_polygon_to_polygon_list

TAG_LIST = [
    "TextRegion",
    "SeparatorRegion",
    "ImageRegion",
    "GraphicRegion",
    "TableRegion",
]


def rescale(args: argparse.Namespace) -> None:
    """Rescale data according to scaling parameter in the provided directory."""
    scale = args.scale**-1 if args.reverse else args.scale
    print(f"scale: {scale}")

    data_path = adjust_path(args.data_path)
    output_path = adjust_path(args.output_path)
    extension = args.extension
    xml = args.xml
    np.random.seed(42)

    if not os.path.exists(f"{output_path}"):
        os.makedirs(f"{output_path}")

    if not os.path.exists(f"{output_path}page/"):
        os.makedirs(f"{output_path}page/")

    if extension is not None:
        image_names = [f[:-4] for f in os.listdir(data_path) if f.endswith(extension)]

        for i in range(args.random + 1):
            number = ""
            if args.random > 0:
                number = "_" + str(i)
            for name in tqdm(
                image_names,
                desc=f"Rescaling images round {i+1}/{args.random +1}",
                unit="image",
            ):
                random_scale = scale
                if args.random > 0:
                    random_scale = 1 + np.random.rand() * (scale - 1)
                    random_scale = (
                        random_scale if np.random.rand() < 0.5 else random_scale**-1
                    )
                    print(f"round {i}: scale {random_scale}")

                image = scale_image(random_scale, data_path, extension, name)  # type: ignore
                image.save(output_path + name + number + extension)

                if xml:
                    bs_data = scale_xml(
                        random_scale, data_path + "page/", name, TAG_LIST
                    )
                    save_xml(bs_data, output_path + "page/", name + number)


def scale_image(scale: int, data_path: str, extension: str, name: str) -> Image.Image:
    """
    Scales one image according to scaling parameter.
    """
    image = Image.open(data_path + name + extension).convert("RGB")
    shape = int(image.size[0] * scale), int(image.size[1] * scale)
    return image.resize(shape, resample=BICUBIC)


def scale_xml(
    scale: int, data_path: str, name: str, tag_list: List[str]
) -> BeautifulSoup:
    """
    Reads xml file, scales all region and text line coordinates and returns the corresponding BeautifulSoup object.

    Args:
        tag_list(list): List of all tags, that should be affected. E.g. 'TextRegion', 'SeparatorRegion'
    """
    with open(f"{data_path + name}.xml", "r", encoding="utf-8") as file:
        data = file.read()

    bs_data = BeautifulSoup(data, "xml")
    page = bs_data.find("Page")
    page.attrs["imageHeight"] = str(int(int(page.attrs["imageHeight"]) * scale))
    page.attrs["imageWidth"] = str(int(int(page.attrs["imageWidth"]) * scale))

    for tag in tag_list:
        regions = bs_data.find_all(tag)
        for region in regions:
            scale_coordinates(region, scale)  # type: ignore
            lines = region.find_all("TextLine")
            for line in lines:
                scale_coordinates(line, scale, True)  # type: ignore

    return bs_data


def scale_coordinates(tag: Tag, scale: int, is_line: bool = False) -> None:
    """
    Extracts coordinates from bs4.Tag object, converts it to an ndarray and scales all coordinates.
    Finally, reconverts coordinates to update the bs4.Tag object.
    """
    polygon = xml_polygon_to_polygon_list(tag.Coords["points"])  # type: ignore
    polygon_ndarray = np.array(polygon, dtype=int).flatten() * scale
    polygon_string = polygon_to_string(polygon_ndarray.tolist(), 1)
    tag.Coords["points"] = polygon_string

    if is_line:
        if tag.Baseline and tag.Baseline["points"]:
            baseline = xml_polygon_to_polygon_list(tag.Baseline["points"])  # type: ignore
            baseline_ndarray = np.array(baseline, dtype=int).flatten() * scale
            baseline_string = polygon_to_string(baseline_ndarray.tolist(), 1)
            tag.Baseline["points"] = baseline_string


# pylint: disable=locally-disabled, duplicate-code
def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="Scale data")
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
        "-e",
        type=str,
        default=None,
        help="Image extension, if this is None, no images will be ignored.",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1,
        help="Scaling factor for images.",
    )
    parser.add_argument(
        "--xml",
        action="store_true",
        help="Convert xml data as well. XML files are expected to be inside a page folder within the data folder.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse scaling, so scale^(-1) is applied.",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=0,
        help="Scale images with uniformly distributed random ratio between scale and scale^(-1). "
        "Provide an integer that determines, how often each image will be scaled.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parameter_args = get_args()
    rescale(parameter_args)
