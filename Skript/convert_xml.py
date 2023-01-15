"""
Script for reading out the Annotations from Transcribus exports
"""
import argparse
import os
import json

import numpy as np

from read_xml import read_transcribus, read_hlna2013
from draw_img import draw_img
from skimage import io
from tqdm import tqdm


INPUT = "../Data/annotationen/"
OUTPUT = "../Data/targets/"


def main():
    read = read_transcribus if args.dataset == 'transcribus' else read_hlna2013
    files = [f[:-4] for f in os.listdir(INPUT) if f.endswith(".xml")]
    for file in tqdm(files):
        annotation = read(f'{INPUT}{file}.xml')
        img = draw_img(annotation)
        io.imsave(f'{OUTPUT}{file}.png', img)

        with open(f'{OUTPUT}{file}.json', 'w') as f:
            json.dump(annotation, f)

        # draw image
        img = draw_img(annotation)

        # save image
        np_save(f"{OUTPUT}{file}", img)
        #img_save(f"{OUTPUT}{file}", img)


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
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', '-d', type=str, default='transcribus', help='select dataset to load '
                                                                                 '(transcribus, HLNA2013)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main()
