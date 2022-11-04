from bs4 import BeautifulSoup
import os
from tqdm import tqdm
import json

"""
Script for reading out the Annotations from HLNA2013 
"""
INPUT = "../Data/annotationen/"
OUTPUT = "../Data/annotationen/"
MISSED = []


def main():
    files = [f[:-4] for f in os.listdir(INPUT) if f.endswith(".xml")]
    for file in tqdm(files):
        annotation = read(f'{INPUT}/{file}.xml')
        with open(f'{OUTPUT}{file}.json', 'w') as f:
            json.dump(annotation, f)

    print(f"Your script missed the following annotations: {set(MISSED)}")


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


if __name__ == '__main__':
    main()
