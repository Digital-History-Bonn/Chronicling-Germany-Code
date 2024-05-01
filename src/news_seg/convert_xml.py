"""
Main Module for converting annotation xml files to numpy images. Also contains backwards converting functions, which
take polygon data and convert it to xml.
"""
import argparse
import os

import numpy as np
from skimage import io
from tqdm import tqdm

from src.news_seg.processing.draw_img_from_polygons import draw_img
from src.news_seg.processing import read_xml

from src.news_seg.utils import draw_prediction


INPUT = "../data/newspaper/annotations/"
OUTPUT = "../data/newspaper/targets/"


def main(parsed_args: argparse.Namespace) -> None:
    """Load xml files and save result image.
    Calls read and draw functions"""
    read = (
        lambda file: read_xml.read_transkribus(path=file, log_path=parsed_args.log_path)  # pylint: disable=unnecessary-lambda-assignment
        if parsed_args.dataset == "transkribus"
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
        img = draw_img(annotation)

        # Debug
        if parsed_args.image_path:
            if not os.path.exists(parsed_args.image_path):
                print(f"creating {parsed_args.image_path}.")
                os.makedirs(parsed_args.image_path)
            draw_prediction(img, f"{parsed_args.image_path}{path}.png")

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
    # pylint: disable=locally-disabled, duplicate-code
    parser = argparse.ArgumentParser(description="creates targets from annotation xmls")
    parser.add_argument(
        "--dataset",
        "-d",
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
    main(args)
