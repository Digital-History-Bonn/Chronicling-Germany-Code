"""
Main Module for converting annotation xml files to numpy images
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from bs4 import BeautifulSoup
from numpy import ndarray
from skimage import io  # type: ignore
from tqdm import tqdm  # type: ignore

from draw_img import draw_img  # type: ignore
from read_xml import read_transcribus, read_hlna2013  # type: ignore

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


def create_xml(segmentations: Dict[int, List[ndarray]], file_name: str, size: Tuple[int, int]) -> BeautifulSoup:
    """
    Creates soup object containing Page Tag and Regions
    :param segmentations: dictionary assigning labels to polygon lists
    :param file_name: image file name
    :param size: image size
    """
    soup = BeautifulSoup()
    soup.append(soup.new_tag("Page", attrs={"imageFilename": file_name, "imageWidth": size[0], "imageHeight": size[1]}))
    page = soup.Page
    for i, (label, segmentation) in enumerate(segmentations.items()):
        for polygon in segmentation:
            region = soup.new_tag(
                    "TextRegion", attrs={"custom": f"readingOrder {{index:{i};}} structure {{type:{label};}}"})
            region.append(soup.new_tag("Coords", attrs={"points": str(polygon)}))
            page.append(region)
    return soup


if __name__ == '__main__':
    args = get_args()
    main()
