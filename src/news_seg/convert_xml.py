"""
Main Module for converting annotation xml files to numpy images. Also contains backwards converting functions, which
take polygon data and convert it to xml.
"""
import argparse
import os
import warnings
from threading import Thread
from typing import List

import numpy as np
from skimage import io
from tqdm import tqdm

from src.news_seg.processing.draw_img_from_polygons import draw_img
from src.news_seg.processing import read_xml

from src.news_seg.utils import draw_prediction, adjust_path

INPUT = "../../data/newspaper/annotations/"
OUTPUT = "../../data/newspaper/targets/"


def main(parsed_args: argparse.Namespace) -> None:
    """Load xml files and save result image.
    Calls read and draw functions"""
    annotations_path = adjust_path(parsed_args.annotations_path)
    output_path = adjust_path(parsed_args.output_path)

    paths = [
        f[:-4] for f in os.listdir(annotations_path) if f.endswith(".xml")
    ]

    if not os.path.exists(output_path):
        print(f"creating {output_path}.")
        os.makedirs(output_path)

    target_paths = [
        f[:-4] for f in os.listdir(output_path) if f.endswith(".npy")
    ]

    threads = []
    for i, path in enumerate(tqdm(paths)):
        threads.append(
            Thread(target=convert_file, args=(path, parsed_args, target_paths)))
        threads[i].start()

        if i % 32 == 0:
            for thread in threads:
                thread.join()
    for thread in threads:
        thread.join()


def convert_file(path: str, parsed_args: argparse.Namespace, target_paths: List[str]) -> None:
    """
    Reads and converts xml data, if that file has not been converted at the start of the script.
    If an image output path is provided, this creates images of the targets and saves them.
    npy files of targets are always saved.
    """
    annotations_path = adjust_path(parsed_args.annotations_path)
    image_path = adjust_path(parsed_args.image_path) if parsed_args.image_path else None
    output_path = adjust_path(parsed_args.output_path)
    log_path = adjust_path(parsed_args.log_path) if parsed_args.log_path else None

    read = (
        # pylint: disable=unnecessary-lambda-assignment
        lambda file: read_xml.read_transkribus(path=file, log_path=log_path)
        if parsed_args.dataset == "transkribus"
        else read_xml.read_hlna2013
    )

    if path in target_paths:
        return
    annotation: dict = read(f"{annotations_path}{path}.xml")  # type: ignore
    if len(annotation) < 1:
        return
    img = draw_img(annotation)

    # Debug
    if image_path:
        if not os.path.exists(image_path):
            print(f"creating {image_path}.")
            os.makedirs(image_path)
        draw_prediction(img, f"{image_path}{path}.png")

    # with open(f"{OUTPUT}{path}.json", "w", encoding="utf-8") as file:
    #     json.dump(annotation, file)

    # save ndarray
    np_save(f"{output_path}{path}", img)


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
    # pylint: disable=locally-disabled, duplicate-code
    parser = argparse.ArgumentParser(description="creates targets from annotation xmls")
    parser.add_argument(
        "--dataset",
        type=str,
        default="transkribus",
        help="select dataset to load " "(transkribus, HLNA2013)",
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
        help="path for output folder",
    )
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        dest="image_path",
        default=None,
        help="path for debug image folder. If no path is supplied, no debug images will be generated.",
    )
    parser.add_argument(
        "--log-path",
        "-l",
        type=str,
        dest="log_path",
        default=None,
        help="path for log file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.image_path:
        warnings.warn("Image output slows down the prediction significantly. "
                      "--image-path should not be activated in production environment.")

    main(args)
