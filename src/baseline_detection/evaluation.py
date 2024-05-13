from typing import Tuple, List
from itertools import product
import re

import torch
from bs4 import BeautifulSoup, PageElement
from shapely import intersection, union
from shapely.geometry import Polygon


def get_tag(textregion: PageElement):
    """
    Returns the tag of the given textregion

    Args:
        textregion: PageElement of Textregion

    Returns:
        Given tag of that Textregion
    """
    desc = textregion['custom']
    match = re.search(r"\{type:.*;\}", desc)
    if match is None:
        return 'UnknownRegion'
    return match.group()[6:-2]


def extract(file: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    with open(file, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    textlines = []
    baselines = []

    text_regions = page.find_all('TextRegion')
    for region in text_regions:
        if get_tag(region) in ['heading', 'article_', 'caption', 'paragraph']:
            text_region = region.find_all('TextLine')
            for text_line in text_region:
                textline = text_line.find('Coords')
                polygon = torch.tensor([tuple(map(int, point.split(','))) for
                                     point in textline['points'].split()])[:, torch.tensor([1, 0])]
                textlines.append(polygon)

                baseline = text_line.find('Baseline')
                line = torch.tensor([tuple(map(int, point.split(','))) for
                                     point in baseline['points'].split()])[:, torch.tensor([1, 0])]
                baselines.append(line)

    return textlines, baselines


def distance(pred: torch.tensor, target: torch.tensor) -> float:
    return 1.0


def iou_matrix(set1: List[Polygon], set2: List[Polygon], threshold: float = .7) -> Tuple[float, float, float]:
    matrix = torch.zeros((len(set1), len(set2)))
    intersects = []
    unions = []

    for a, b in product(set1, set2):
        intersects.append(intersection(a, b).area)
        unions.append(union(a, b).area)

    ious = (torch.tensor(intersects) / torch.tensor(unions)).reshape(len(set1), len(set2))

    pred_iuo = ious.amax(dim=1)
    target_iuo = ious.amax(dim=0)

    tp = torch.sum(pred_iuo >= threshold)
    fp = len(matrix) - tp
    fn = torch.sum(target_iuo < threshold)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def evaluation(prediction_file: str, ground_truth_file: str):
    """
    Evaluates the baseline detection.

    Args:
        prediction_file: Path to the prediction file.
        ground_truth_file: Path to the ground truth file.
    """
    pred_polygons, pred_baselines = extract(prediction_file)
    truth_polygons, truth_baselines = extract(ground_truth_file)

    set1 = [Polygon(poly) for poly in pred_polygons]
    set2 = [Polygon(poly) for poly in truth_polygons]

    precision, recall, f1_score = iou_matrix(set1, set2)

    # dist = distance(pred_baselines, truth_baselines)


if __name__ == '__main__':
    evaluation(prediction_file='../../data/pero_lines_bonn_regions/Koelnische Zeitung 1866.06-1866.09 - 0046.xml',
               ground_truth_file='../../data/pero_lines_bonn_regions/Koelnische Zeitung 1866.06-1866.09 - 0046.xml')