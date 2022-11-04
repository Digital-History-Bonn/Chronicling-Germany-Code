import numpy as np
from bs4 import BeautifulSoup
import re
from skimage import draw
from skimage import io
import os
from tqdm import tqdm
import json

"""
Script for reading out the Annotations from Transcribus exports 
"""
label_assignments = {"article": 1, "heading": 2, "header": 3, "table": 4, "separator_vertical": 5, "UnknownRegion": 6,
                     "caption": 7, "separator_short": 8, "separator_horizontal": 9, "TextLine": 10}
INPUT = "annotations/"
OUTPUT = "targets/"
MISSED = []


def main():
    files = [f[:-4] for f in os.listdir(INPUT) if f.endswith(".xml")]
    for file in tqdm(files):
        annotation = read(f'annotations/{file}.xml')
        img = draw_img(annotation)
        io.imsave(f'{OUTPUT}{file}.png', img)
        with open(f'{OUTPUT}{file}.json', 'w') as f:
            json.dump(annotation, f)

    print(set(MISSED))


def read(path):
    """
    reads xml file and returns dictionary containing annotations
    :param path: path to file
    :return: dictionary {height: , width: , tags: {tag_name_1: [], tag_name_2: [], ...}}
    """
    with open(path, 'r') as f:
        data = f.read()

    Bs_data = BeautifulSoup(data, "xml")
    tags_dict = {'TextLine': []}

    tags_dict = find_regions(Bs_data, 'TextRegion', True, 'TextLine', tags_dict)
    tags_dict = find_regions(Bs_data, 'SeparatorRegion', False, '', tags_dict)

    page = Bs_data.find('Page')

    return {"size": (int(page['imageWidth']), int(page['imageHeight'])), 'tags': tags_dict}


def find_regions(data, tag, search_children, child_tag, tags_dict):
    """
    returns dictionary with all coordinates of specified regions
    :param data: BeautifulSoup xml data
    :param tag: tag to be found
    :param search_children: only True if there are children to be included
    :param child_tag: children tag to be found
    :param tags_dict: dictionary to contain region data
    :return: tags: {tag_name_1: [], tag_name_2: [], ...}
    """
    regions = data.find_all(tag)
    for region in regions:
        type = re.search('readingOrder \{index:(.+?);} structure \{type:(.+?);}', region['custom'])
        if type is None:
            type = "UnknownRegion"
        else:
            type = type.group(2)
        if type not in tags_dict:
            tags_dict[type] = []
        tags_dict[type].append([pair.split(',') for pair in region.Coords["points"].split()])
        if search_children:
            lines = region.find_all(child_tag)
            if child_tag not in tags_dict:
                tags_dict[child_tag] = []
            for line in lines:
                tags_dict[child_tag].append([pair.split(',') for pair in line.Coords["points"].split()])
    return tags_dict


def draw_img(annotation):
    """
    draws an image with the information from the read-function
    :param annotation: dict with information
    :return: ndarray
    """

    x, y = annotation['size']
    img = np.zeros((x, y))

    for key, label in label_assignments.items():
        if key in annotation['tags'].keys():
            for polygon in annotation['tags'][key]:
                img = draw_polygon(img, polygon, label=6)

    return img


def draw_polygon(img, polygon, label=1):
    polygon = np.array(polygon, dtype=int).T
    rr, cc = draw.polygon(polygon[1], polygon[0])
    img[rr, cc] = label

    return img


if __name__ == '__main__':
    main()
