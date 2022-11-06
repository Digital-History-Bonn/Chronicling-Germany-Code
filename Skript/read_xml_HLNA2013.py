from bs4 import BeautifulSoup
import os
import sys
from tqdm import tqdm
import json

"""
Script for reading out the Annotations from HLNA2013 
"""
INPUT = "../Data/annotationen/"
OUTPUT = "../Data/annotationen/"


def main(data=INPUT, output=OUTPUT):
    files = [f[:-4] for f in os.listdir(data) if f.endswith(".xml")]
    for file in tqdm(files):
        annotation = read(f'{data}/{file}.xml')
        with open(f'{output}{file}.json', 'w') as f:
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
    bs_data = BeautifulSoup(data, "xml")
    annotation['size'] = (int(bs_data.find('Page').get('imageHeight')), int(bs_data.find('Page').get('imageWidth')))
    annotation['tags'] = {}

    text_regions = bs_data.find_all('TextRegion')
    separator_regions = bs_data.find_all('SeparatorRegion')
    table_regions = bs_data.find_all('TableRegion')

    # get coordinates of all TextRegions
    paragraphs = []
    headings = []
    header = []
    unknown_region = []
    for sep in text_regions:
        coord = sep.find('Coords')
        if sep.get('type') == 'heading':
            headings.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])
        elif sep.get('type') == 'paragraph':
            paragraphs.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])
        elif sep.get('type') == 'header':
            header.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])
        else:
            unknown_region.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])

    annotation['tags']['article'] = paragraphs
    annotation['tags']['heading'] = headings
    annotation['tags']['header'] = header
    annotation['tags']['UnknownRegion'] = unknown_region

    # get coordinates of all seperators
    separator = []
    for sep in separator_regions:
        coord = sep.find('Coords')
        separator.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])

    annotation['tags']['separator_vertical'] = separator

    # get coordinates of all Tables
    tabels = []
    for table in table_regions:
        coord = table.find('Coords')
        tabels.append([(int(p.get('x')), int(p.get('y'))) for p in coord.find_all('Point')])

    annotation['tags']['table'] = tabels

    return annotation


if __name__ == '__main__':
    assert len(sys.argv) == 3, "function needs 2 arguments."
    data = sys.argv[1]
    output = sys.argv[2]
    main(data=data, output=output)
