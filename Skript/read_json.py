from draw_img import draw_img

import numpy as np
from skimage import io
from tqdm import tqdm
import sys
import os
import json


INPUT = "../Data/annotationen/"
OUTPUT = "../Data/Targets/"
MISSED = []


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


def main(mode: str = 'image', data: str = INPUT, output: str = OUTPUT):
    modes = {'numpy': np_save, 'image': img_save}
    assert mode in modes.keys(), f"mode {mode} not in possible modes (\"numpy\", \"image\")"

    # list all json files in INPUT-folder
    files = [f[:-5] for f in os.listdir(data) if f.endswith(".json")]
    for file in tqdm(files):
        # open all annotation jsons
        with open(f'{data}{file}.json', 'r') as f:
            annotation = json.load(f)

        # draw image
        img = draw_img(annotation)

        # save image
        modes[mode](f"{output}{file}", img)


if __name__ == '__main__':
    assert len(sys.argv) == 4, "function needs 3 arguments."
    mode = sys.argv[1]
    input = sys.argv[2]
    output = sys.argv[3]
    main(mode='numpy', data=input, output=output)
