"""
Main Module for converting annotation xml files to numpy images. Also contains backwards converting functions, which
take polygon data and convert it to xml.
"""
import argparse
import json
import os
import warnings
from multiprocessing import Process, Queue
from time import sleep
from typing import List

import numpy as np
from skimage import io
from tqdm import tqdm

from src.layout_segmentation.processing.draw_img_from_polygons import draw_img
from src.layout_segmentation.processing import read_xml

from src.layout_segmentation.utils import draw_prediction, adjust_path

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


    path_queue: Queue = Queue()
    processes = [Process(target=convert_file, args=(path_queue, parsed_args, target_paths)) for _ in range(32)]
    for process in processes:
        process.start()
    for path in tqdm(paths, desc="Put paths in queue"):
        path_queue.put((path, False))
    total = len(paths)
    with tqdm(total=path_queue.qsize(), desc="Page converting", unit="pages") as pbar:
        while not path_queue.empty():
            pbar.n = total - path_queue.qsize()
            pbar.refresh()
            sleep(1)
    for _ in processes:
        path_queue.put(("", True))
    for process in tqdm(processes, desc="Waiting for processes to end"):
        process.join()


def convert_file(path_queue: Queue, parsed_args: argparse.Namespace, target_paths: List[str]) -> None:
    """
    Reads and converts xml data, if that file has not been converted at the start of the script.
    If an image output path is provided, this creates images of the targets and saves them.
    npy files of targets are always saved.
    """
    annotations_path = adjust_path(parsed_args.annotations_path)
    image_path = adjust_path(parsed_args.image_path) if parsed_args.image_path else None
    output_path = adjust_path(parsed_args.output_path)

    while True:
        path, done = path_queue.get()
        if done:
            break
        read = (
            # pylint: disable=unnecessary-lambda-assignment
            lambda file: read_xml.read_transkribus(path=file, log=parsed_args.log)
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

        if parsed_args.json:
            with open(f"{OUTPUT}{path}.json", "w", encoding="utf-8") as file:
                json.dump(annotation, file)

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
        "--log",
        "-l",
        action="store_true",
        dest="log",
        help="Activates command line unkown regions logging. This reports all unkown regions and region type "
             "mismatches for each file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Activates json dump of polygon dictionary.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.image_path:
        warnings.warn("Image output slows down the prediction significantly. "
                      "--image-path should not be activated in production environment.")

    main(args)
