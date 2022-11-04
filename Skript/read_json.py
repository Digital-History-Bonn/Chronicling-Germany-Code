from draw_img import draw_img

import numpy as np
from skimage import io
from tqdm import tqdm
import os
import json


INPUT = "../Data/annotationen/"
OUTPUT = "../Data/Targets/"
MISSED = []


def np_save(outfile, img):
    """
    saves given image in outfile.npy
    :param outfile: name of the file without ending
    :param img: numpy array to save
    """
    np.save(f"{OUTPUT}{outfile}.npy", img)


def img_save(outfile, img):
    """
    saves given as outfile.png
    :param outfile: name of the file without ending
    :param img: numpy array to save
    """
    io.imsave(f'{OUTPUT}{outfile}.png', img)


def main(mode='numpy'):
    files = [f[:-5] for f in os.listdir(INPUT) if f.endswith(".json")]
    for file in tqdm(files):
        with open(f'{INPUT}{file}.json', 'r') as f:
            annotation = json.load(f)
            img = draw_img(annotation)

            {'numpy': np_save, 'image': img_save}[mode](file, img)


if __name__ == '__main__':
    main()