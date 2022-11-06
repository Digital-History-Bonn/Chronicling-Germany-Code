"""
Script for reading out the Annotations from Transcribus exports
"""
import re
import os
import json
from draw_img import draw_img
from bs4 import BeautifulSoup
from skimage import io
from tqdm import tqdm


INPUT = "../Data/annotationen/"
OUTPUT = "../Data/targets/"


def main():
    files = [f[:-4] for f in os.listdir(INPUT) if f.endswith(".xml")]
    for file in tqdm(files):
        annotation = read(f'annotations/{file}.xml')
        img = draw_img(annotation)
        io.imsave(f'{OUTPUT}{file}.png', img)
        with open(f'{OUTPUT}{file}.json', 'w') as f:
            json.dump(annotation, f)


def read(path):
    """
    reads xml file and returns dictionary containing annotations
    :param path: path to file
    :return: dictionary {height: , width: , tags: {tag_name_1: [], tag_name_2: [], ...}}
    """
    with open(path, 'r') as f:
        data = f.read()

    bs_data = BeautifulSoup(data, "xml")
    tags_dict = {'TextLine': []}

    tags_dict = find_regions(bs_data, 'TextRegion', True, 'TextLine', tags_dict)
    tags_dict = find_regions(bs_data, 'SeparatorRegion', False, '', tags_dict)

    page = bs_data.find('Page')

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
        region_type = re.search("readingOrder \{index:(.+?);} structure \{type:(.+?);}", region['custom'])
        if region_type is None:
            region_type = "UnknownRegion"
        else:
            region_type = region_type.group(2)
        if region_type not in tags_dict:
            tags_dict[region_type] = []
        tags_dict[region_type].append([pair.split(',') for pair in region.Coords["points"].split()])
        if search_children:
            lines = region.find_all(child_tag)
            if child_tag not in tags_dict:
                tags_dict[child_tag] = []
            for line in lines:
                tags_dict[child_tag].append([pair.split(',') for pair in line.Coords["points"].split()])
    return tags_dict


if __name__ == '__main__':
    main()
