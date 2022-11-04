from skimage import draw
import numpy as np


def draw_img(annotation, elements=None):
    """
    draws an image with the information from the read-function
    :param annotation: dict with information
    :param elements: dict of elements with their labels to draw on img
    :return: ndarray
    """

    if elements is None:
        elements = {'footnote': 1, 'paragraphs': 2, 'header': 3, 'tables': 4, 'separator': 5}

    x, y = annotation['size']
    img = np.zeros((x, y))

    if 'footnote' in elements and 'footnote' in annotation.keys():
        for polygon in annotation['footnote']:
            img = draw_polygon(img, polygon, label=6)

    if 'paragraphs' in elements and 'paragraphs' in annotation.keys():
        for polygon in annotation['paragraphs']:
            img = draw_polygon(img, polygon, label=1)

    if 'headings' in elements and 'headings' in annotation.keys():
        for polygon in annotation['headings']:
            img = draw_polygon(img, polygon, label=2)

    if 'header' in elements and 'header' in annotation.keys():
        for polygon in annotation['header']:
            img = draw_polygon(img, polygon, label=3)

    if 'tables' in elements and annotation.keys():
        for polygon in annotation['tables']:
            img = draw_polygon(img, polygon, label=4)

    if 'separator' in elements and 'separator' in annotation.keys():
        for polygon in annotation['separator']:
            img = draw_polygon(img, polygon, label=5)

    return img


def draw_polygon(img, polygon, label=1):
    """
    draws a polygon on the given image
    :param img: numpy array
    :param polygon: list of dots
    :param label: label of the polygon
    """
    polygon = np.array(polygon).T
    rr, cc = draw.polygon(polygon[1], polygon[0])
    img[rr, cc] = label

    return img