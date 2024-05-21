"""Evaluation script for baseline detection."""
from typing import Tuple, List
from itertools import product

import torch
from bs4 import BeautifulSoup
from shapely import intersection, union
from shapely.geometry import Polygon

from src.baseline_detection.utils import get_tag


def extract(file_path: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Extracts predicted textlines and baseline from the xml file.

    Args:
        file_path: path to the xml file.

    Returns:
        textlines: List of predicted textlines.
        baselines: List of baseline predicted textlines.
    """
    with open(file_path, "r", encoding="utf-8") as file:
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
                                        point in textline['points'].split()])
                polygon = polygon[:, torch.tensor([1, 0])]
                textlines.append(polygon)

                baseline = text_line.find('Baseline')
                line = torch.tensor([tuple(map(int, point.split(','))) for
                                     point in baseline['points'].split()])[:, torch.tensor([1, 0])]
                baselines.append(line)

    return textlines, baselines


def distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Not implemented yet.

    Args:
        pred: Predicted baselines.
        target: Target baselines.

    Returns:
        Distance between the predicted and target baselines.
    """
    return 1.0


def textline_detection_metrics(prediction: List[Polygon],
                               target: List[Polygon],
                               threshold: float = .7) -> Tuple[float, float, float]:
    """
    Calcs precision, recall, F1 score for textline polygons.

    Textline is considered as detected if IoU is above threshold.
    Also textline predictions are considerd as correct if IoU is above threshold.

    Args:
        prediction: List of predicted textlines.
        target: List of ground truth textlines.
        threshold: Threshold for IoU.

    Returns:
        precision, recall, F1 score
    """
    matrix = torch.zeros((len(prediction), len(target)))
    intersects = []
    unions = []

    for pred, tar in product(prediction, target):
        intersects.append(intersection(pred, tar).area)
        unions.append(union(pred, tar).area)

    ious = (torch.tensor(intersects) / torch.tensor(unions)).reshape(len(prediction), len(target))

    pred_iuo = ious.amax(dim=1)
    target_iuo = ious.amax(dim=0)

    tp = torch.sum(pred_iuo >= threshold).item()
    fp = len(matrix) - tp
    fn = torch.sum(target_iuo < threshold).item()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def evaluation(prediction_file: str, ground_truth_file: str) -> None:
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

    precision, recall, f1_score = textline_detection_metrics(set1, set2)

    # dist = distance(pred_baselines, truth_baselines)


if __name__ == '__main__':
    evaluation(prediction_file='../../data/pero_lines_bonn_regions/'
                               'Koelnische Zeitung 1866.06-1866.09 - 0046.xml',
               ground_truth_file='../../data/pero_lines_bonn_regions/'
                                 'Koelnische Zeitung 1866.06-1866.09 - 0046.xml')
