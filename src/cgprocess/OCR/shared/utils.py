"""Utility functions for OCR."""

import argparse
import os
import random
from multiprocessing import Queue
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import yaml
from bs4 import BeautifulSoup
from PIL import Image, ImageOps
from skimage import io

from src.cgprocess.OCR.shared.tokenizer import OCRTokenizer
from src.cgprocess.shared.utils import get_bbox, xml_polygon_to_polygon_list


def pad_xml(soup: BeautifulSoup, pad_value: int = 10) -> BeautifulSoup:
    """
    Pads the coordinates of all elements in given BeautifulSoup soup.

    Args:
        soup: BeautifulSoup with annotations.
        pad_value: Value for padding

    Returns:
        Padded BeautifulSoup soup
    """
    # Find all elements with 'points' attribute
    elements_with_points = soup.find_all(attrs={"points": True})

    for element in elements_with_points:
        points: str = element["points"]  # type: ignore
        padded_points = pad_points(points, pad_value)
        element["points"] = padded_points  # type: ignore

    return soup


def pad_points(points: str, pad_value: int = 10) -> str:
    """
    Pads each point in the points string by the given pad value.

    Args:
        points: A string of points in the format 'x1,y1 x2,y2 ...'.
        pad_value: The padding value (int) to apply to each point.

    Returns:
        A string of padded points.
    """
    points_list = points.split()
    padded_points_list = []

    for point in points_list:
        x, y = map(int, point.split(","))
        padded_x = x + pad_value
        padded_y = y + pad_value
        padded_points_list.append(f"{padded_x},{padded_y}")

    return " ".join(padded_points_list)


def pad_image(image: Image.Image, pad: int = 10) -> Image.Image:
    """
    Pads the given PIL Image with a specified number of pixels on each side.

    Args:
        image (Image.Image): The input PIL Image to be padded.
        pad (int): The number of pixels to pad on each side. Default is 10.

    Returns:
        Image.Image: The padded PIL Image.
    """
    return ImageOps.expand(image, border=(pad, pad, pad, pad), fill=0)


def adjust_path(path: str) -> str:
    """
    Make sure, there is no slash at the end of a (folder) spath string.

    Args:
        path: String representation of path

    Returns:
        path without ending '/'
    """
    return path if not path or path[-1] != "/" else path[:-1]


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def line_has_text(line: BeautifulSoup) -> bool:
    """
    Checks if line has text in it.

    Args:
        line: BeautifulSoup object representing line

    Returns:
        True if the line has TextEquiv and Unicode xml tags"""
    return bool(
        line.TextEquiv
        and line.TextEquiv.Unicode
        and len(line.TextEquiv.Unicode.contents)
    )


def load_image(image_path: str) -> torch.Tensor:
    """
    Loads an image and ensures it has the right dimensions.

    Args:
        image_path: path to image

    Returns:
        torch tensor of shape (H, W, C) with values in the range [0, 1]
    """
    # pylint: disable=duplicate-code
    image = torch.from_numpy(io.imread(image_path))
    # image is black and white
    if image.dim() == 2:
        return image[None, :, :].repeat(3, 1, 1) / 256

    # image has channels last
    if image.shape[-1] == 3:
        return image.permute(2, 0, 1) / 256

    # image has alpha channel and channel last
    if image.shape[-1] == 4:
        return image[:, :, :3].permute(2, 0, 1) / 256

    return image / 256


def create_path_queue(
    annotations: List[str], args: argparse.Namespace, images: List[str]
) -> Queue:
    """
    Create path queue for OCR prediction containing image, annotation (baseline) and ouput path.
    Elements are required to have the image path at
    index 0 and the bool variable for terminating processes at index -1.
    :param annotations: list of annotation paths
    :param images: list of image paths
    """
    path_queue: Queue = Queue()
    for image_path, annotation_path in zip(images, annotations):
        output_path = f"{args.output}/{os.path.basename(annotation_path)}"
        path_queue.put((image_path, annotation_path, output_path, False))
    return path_queue


def init_model(model: Any) -> Any:
    """Init function for compatibility with the MPPredictor handling baseline and layout predictions as well."""
    return model


def read_xml(xml_path: str) -> Tuple[List[torch.Tensor], List[str], List[torch.Tensor]]:
    """
    Reads the xml files.
    Args:
        xml_path: path to xml file with annotations.

    Returns:
        bboxes: bounding boxes text lines.
        texts: text of text lines.
    """
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, "xml")
    page = soup.find("Page")
    bboxes = []
    texts = []
    region_polygons = []

    text_lines = page.find_all("TextLine")  # type: ignore
    for line in text_lines:
        if line_has_text(line):
            region_polygon = torch.tensor(
                xml_polygon_to_polygon_list(line.Coords["points"])
            )
            region_polygons.append(region_polygon)
            bboxes.append(torch.tensor(get_bbox(region_polygon)))
            texts.append(line.find("Unicode").text)

    return bboxes, texts, region_polygons


def create_unicode_alphabet(length: int) -> List[str]:
    """
    Creates alphabet with Unicode characters.
    Args:
        length: number of Unicode characters in alphabet
    """
    result = []
    for i in range(length):
        result.append(chr(i))
    return ["<PAD>", "<START>", "<NAN>", "<END>"] + result


def load_cfg(config_path: Path) -> dict:
    """Load yml config from supplied path."""
    with open(config_path, "r", encoding="utf-8") as file:
        cfg: dict = yaml.safe_load(file)
    return cfg


def init_tokenizer(cfg: dict) -> OCRTokenizer:
    """Initialize tokenizer by creating the vocabulary and setting config accordingly."""
    unicode_alphabet = create_unicode_alphabet(cfg["vocabulary"]["unicode"])
    custom_alphabet = cfg["vocabulary"].get("custom", [])
    for char in custom_alphabet:
        if char not in unicode_alphabet:
            unicode_alphabet.append(char)
    cfg["vocabulary"]["size"] = len(unicode_alphabet)
    return OCRTokenizer(unicode_alphabet, **cfg["tokenizer"])
