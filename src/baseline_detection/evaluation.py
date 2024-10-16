"""Evaluation script for baseline detection."""
import argparse
import glob
import os.path
from typing import Tuple, List
from itertools import product

import torch
from bs4 import BeautifulSoup
from shapely.geometry import Polygon
from tqdm import tqdm

from src.baseline_detection.class_config import TEXT_CLASSES
from src.baseline_detection.utils import get_tag, adjust_path


def extract_textlines(file_path: str) -> List[torch.Tensor]:
    """
    Extracts predicted textlines and baseline from the xml file.

    Args:
        file_path: path to the xml file.

    Returns:
        textlines: List of predicted textlines.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    textlines = []

    text_regions = page.find_all('TextRegion')
    for region in text_regions:
        if get_tag(region) in TEXT_CLASSES:
            text_region = region.find_all('TextLine')
            for text_line in text_region:
                textline = text_line.find('Coords')
                polygon = torch.tensor([tuple(map(int, point.split(','))) for
                                        point in textline['points'].split()])
                polygon = polygon[:, torch.tensor([1, 0])]
                textlines.append(polygon)

    return textlines


def textline_detection_metrics(prediction: List[Polygon],
                               target: List[Polygon],
                               threshold: float = .7) -> Tuple[
    int, int, int, float, float, float]:
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
    intersects = []
    unions = []

    for pred, tar in product(prediction, target):
        intersects.append(pred.intersection(tar).area)
        unions.append(pred.union(tar).area)

    ious = (torch.tensor(intersects) / torch.tensor(unions)).reshape(len(prediction), len(target))

    results = calc_metrics(ious, threshold=threshold)

    return results


def calc_metrics(ious: torch.Tensor,
                 threshold: float = .7) -> Tuple[int, int, int, float, float, float]:
    """
    calculates precision, recall and f1_score for iou matrix.

    Args:
        ious: matrix with intersection over union of textline polygons

    Returns:
        precision, recall and f1_score
    """
    # pylint: disable=duplicate-code
    tp = 0
    while True:
        max_index = torch.argmax(ious)
        max_row = max_index // ious.size(1)
        max_col = max_index % ious.size(1)
        max_value = ious[max_row, max_col]

        if max_value < threshold:
            break

        ious[max_row, :] = 0
        ious[:, max_col] = 0

        tp += 1

    fp = ious.shape[0] - tp
    fn = ious.shape[1] - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

    return tp, fp, fn, precision, recall, f1_score


def evaluation(prediction_file: str, ground_truth_file: str) -> Tuple[
    int, int, int, float, float, float]:
    """
    Evaluates the baseline detection.

    Args:
        prediction_file: Path to the prediction file.
        ground_truth_file: Path to the ground truth file.
    """
    pred = extract_textlines(prediction_file)
    ground_truth = extract_textlines(ground_truth_file)

    pred_polygons = [Polygon(poly) for poly in pred]
    truth_polygons = [Polygon(poly) for poly in ground_truth]

    results = textline_detection_metrics(pred_polygons, truth_polygons)

    return results


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Baseline evaluation")
    parser.add_argument(
        "--prediction_dir",
        "-p",
        type=str,
        default=None,
        help="path for folder with prediction xml files."
    )

    parser.add_argument(
        "--ground_truth_dir",
        "-g",
        type=str,
        default=None,
        help="path for folder with ground truth xml files."
    )

    return parser.parse_args()


def main() -> None:
    """Evaluates baseline predicts."""
    args = get_args()

    pred_dir = adjust_path(args.prediction_dir)
    target_dir = adjust_path(args.ground_truth_dir)

    targets = list(glob.glob(f"{target_dir}/*.xml"))
    predictions = [f"{pred_dir}/{os.path.basename(x)}" for x in targets]

    all_tp, all_fp, all_fn = 0, 0, 0
    precisions, recalls, f1_scores = [], [], []

    for target, prediction in tqdm(zip(targets, predictions), total=len(targets),
                                   desc='evaluation'):
        tp, fp, fn, precision, recall, f1_score = evaluation(target, prediction)

        print(f"{target=}\n"
              f"\t{tp=}\n"
              f"\t{fp=}\n"
              f"\t{fn=}\n"
              f"\t{precision=}\n"
              f"\t{recall=}\n"
              f"\t{f1_score=}\n\n")

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

        all_tp += tp
        all_fp += fp
        all_fn += fn

    print(f"average precision: {torch.mean(torch.tensor([precisions]))}"
          f" ({torch.median(torch.tensor([precisions]))})"
          f" +- {torch.std(torch.tensor([precisions]))},"
          f" min: {torch.min(torch.tensor([precisions]))},"
          f" max: {torch.max(torch.tensor([precisions]))}")
    print(f"average recall: {torch.mean(torch.tensor([recalls]))}"
          f" ({torch.median(torch.tensor([recalls]))})"
          f" +- {torch.std(torch.tensor([recalls]))},"
          f" min: {torch.min(torch.tensor([recalls]))},"
          f" max: {torch.max(torch.tensor([recalls]))}")
    print(f"average f1_score: {torch.mean(torch.tensor([f1_scores]))}"
          f" ({torch.median(torch.tensor([f1_scores]))})"
          f" +- {torch.std(torch.tensor([f1_scores]))},"
          f" min: {torch.min(torch.tensor([f1_scores]))},"
          f" max: {torch.max(torch.tensor([f1_scores]))}\n")

    all_precision = all_tp / (all_tp + all_fp)
    all_recall = all_tp / (all_tp + all_fn)
    print(f"overall precision: {all_precision}")
    print(f"overall recall: {all_recall}")
    print(f"overall f1_score: {2 * (all_precision * all_recall) / (all_precision + all_recall)}")


if __name__ == '__main__':
    main()
