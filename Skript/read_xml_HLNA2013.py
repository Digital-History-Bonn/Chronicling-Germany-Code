from bs4 import BeautifulSoup
import os
from tqdm import tqdm
import json

"""
Script for reading out the Annotations from HLNA2013 
"""
INPUT = "../Data/annotationen/"
OUTPUT = "../Data/annotationen/"


def main():
    files = [f[:-4] for f in os.listdir(INPUT) if f.endswith(".xml")]
    for file in tqdm(files):
        annotation = read(f'{INPUT}/{file}.xml')
        with open(f'{OUTPUT}{file}.json', 'w') as f:
            json.dump(annotation, f)


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
    annotation['tags'] = {}

    TextRegions = Bs_data.find_all('TextRegion')
    SeparatorRegions = Bs_data.find_all('SeparatorRegion')
    TableRegions = Bs_data.find_all('TableRegion')

    # get coordinates of all TextRegions
    paragraphs = []
    headings = []
    header = []
    UnknownRegion = []
    for sep in TextRegions:
        coord = sep.find('Coords')
        if sep.get('type') == 'heading':
            headings.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])
        elif sep.get('type') == 'paragraph':
            paragraphs.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])
        elif sep.get('type') == 'header':
            header.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])
        else:
            UnknownRegion.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])

    annotation['tags']['article'] = paragraphs
    annotation['tags']['heading'] = headings
    annotation['tags']['header'] = header
    annotation['tags']['UnknownRegion'] = UnknownRegion

    # get coordinates of all seperators
    separator = []
    for sep in SeparatorRegions:
        coord = sep.find('Coords')
        separator.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])

    annotation['tags']['separator_vertical'] = separator

    # get coordinates of all Tables
    tabels = []
    for table in TableRegions:
        coord = table.find('Coords')
        tabels.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])

    annotation['tags']['table'] = tabels

    return annotation


if __name__ == '__main__':
    main()
