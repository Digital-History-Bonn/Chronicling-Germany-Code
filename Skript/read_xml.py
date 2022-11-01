from bs4 import BeautifulSoup
import numpy as np
from skimage import io
from skimage import draw
import os
from tqdm import tqdm
import json

"""
Script for reading out the Annotations from HLNA2013 
"""
INPUT = "Annotationen/"
OUTPUT = "Targets/"
MISSED = []


def main():
    files = [f[:-4] for f in os.listdir(INPUT) if f.endswith(".xml")]
    for file in tqdm(files):
        annotation = read(f'Annotationen/{file}.xml')
        img = draw_img(annotation)
        io.imsave(f'{OUTPUT}{file}.png', img)
        with open(f'{OUTPUT}{file}.json', 'w') as f:
            json.dump(annotation, f)

    print(set(MISSED))


def read(path):
    """
    reads xml file and returns important information in dict
    :param path: path to file
    :return: dict with important information
    """
    annotation = {}
    with open(path, 'r') as f:
        data = f.read()

    # read xml
    Bs_data = BeautifulSoup(data, "xml")
    annotation['size'] = (int(Bs_data.find('Page').get('imageHeight')), int(Bs_data.find('Page').get('imageWidth')))

    TextRegions = Bs_data.find_all('TextRegion')
    SeparatorRegions = Bs_data.find_all('SeparatorRegion')
    TableRegions = Bs_data.find_all('TableRegion')

    # get coordinates of all TextRegions
    paragraphs = []
    headings = []
    header = []
    footnote = []
    for sep in TextRegions:
        coord = sep.find('Coords')
        if sep.get('type') == 'heading':
            headings.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])
        elif sep.get('type') == 'paragraph':
            paragraphs.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])
        elif sep.get('type') == 'header':
            header.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])
        elif sep.get('type') == 'footnote':
            footnote.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])
        else:
            MISSED.append(sep.get('type'))

    annotation['paragraphs'] = paragraphs
    annotation['headings'] = headings
    annotation['header'] = header
    annotation['footnote'] = footnote

    # get coordinates of all seperators
    separator = []
    for sep in SeparatorRegions:
        coord = sep.find('Coords')
        separator.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])

    annotation['separator'] = separator

    # get coordinates of all Tables
    tabels = []
    for table in TableRegions:
        coord = table.find('Coords')
        tabels.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])

    annotation['tables'] = tabels

    return annotation


def draw_img(annotation):
    """
    draws an image with the information from the read-function
    :param annotation: dict with information
    :return: ndarray
    """

    x, y = annotation['size']
    img = np.zeros((x, y))

    if 'footnote' in annotation.keys():
        for polygon in annotation['footnote']:
            img = draw_polygon(img, polygon, label=6)

    if 'paragraphs' in annotation.keys():
        for polygon in annotation['paragraphs']:
            img = draw_polygon(img, polygon, label=1)

    if 'headings' in annotation.keys():
        for polygon in annotation['headings']:
            img = draw_polygon(img, polygon, label=2)

    if 'header' in annotation.keys():
        for polygon in annotation['header']:
            img = draw_polygon(img, polygon, label=3)

    if 'tables' in annotation.keys():
        for polygon in annotation['tables']:
            img = draw_polygon(img, polygon, label=4)

    if 'separator' in annotation.keys():
        for polygon in annotation['separator']:
            img = draw_polygon(img, polygon, label=5)

    return img


def draw_polygon(img, polygon, label=1):
    polygon = np.array(polygon).T
    rr, cc = draw.polygon(polygon[1], polygon[0])
    img[rr, cc] = label

    return img


if __name__ == '__main__':
    main()