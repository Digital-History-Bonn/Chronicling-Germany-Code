"""
Main Module for converting annotation xml files to numpy images
"""
import argparse
import json
import os

import numpy as np
from skimage import io  # type: ignore
from tqdm import tqdm  # type: ignore

import draw_img
import read_xml

INPUT = "../DataBonn/Annotations/"
OUTPUT = "../DataBonn/targets/"


def main():
    """Load xml files and save result image.
    Calls read and draw functions"""
    read = read_xml.read_transcribus if args.dataset == 'transcribus' else read_xml.read_hlna2013
    paths = [f[:-4] for f in os.listdir(INPUT) if f.endswith(".xml")]
    for path in tqdm(paths):
        annotation = read(f'{INPUT}{path}.xml')
        img = draw_img.draw_img(annotation)
        io.imsave(f'{OUTPUT}{path}.png', img/10)

        with open(f'{OUTPUT}{path}.json', 'w', encoding="utf-8") as file:
            json.dump(annotation, file)

        # draw image
        img = draw_img.draw_img(annotation)

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


if __name__ == '__main__':
    args = get_args()
    main()
