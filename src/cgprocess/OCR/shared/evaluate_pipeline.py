"""Evaluation script for baseline detection."""
import argparse
import glob
import json
import os.path
import time
from typing import Tuple, List
from itertools import product

import numpy as np
import torch
from bs4 import BeautifulSoup
from shapely.geometry import Polygon
from tqdm import tqdm

from src.cgprocess.OCR.shared.evaluate_ocr import levensthein_distance, calculate_ratio
from src.cgprocess.baseline_detection.class_config import TEXT_CLASSES
from src.cgprocess.baseline_detection.utils import get_tag, adjust_path
from src.cgprocess.layout_segmentation.processing.read_xml import xml_polygon_to_polygon_list


# pylint: disable=duplicate-code
def extract_textlines(file_path: str) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Extracts predicted textlines and baseline from the xml file.

    Args:
        file_path: path to the xml file.

    Returns:
        textlines: List of predicted textlines.
        texts: List of texts of the predicted textlines.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    textlines = []
    texts = []

    text_regions = page.find_all('TextRegion')
    for region in text_regions:
        if get_tag(region) in TEXT_CLASSES:
            text_region = region.find_all('TextLine')
            for text_line in text_region:
                polygon = torch.tensor(xml_polygon_to_polygon_list(text_line.Coords["points"]))
                polygon = polygon[:, torch.tensor([1, 0])]
                textlines.append(polygon)

                textequiv = text_line.find('TextEquiv')
                texts.append(textequiv.find('Unicode').text)

    return textlines, texts


def matching(predictions: List[Polygon],
             targets: List[Polygon]) -> List[Tuple[int, int]]:
    """
    Calcs precision, recall, F1 score for textline polygons.

    Textline is considered as detected if IoU is above threshold.
    Also textline predictions are considerd as correct if IoU is above threshold.

    Args:
        predictions: List of predicted textlines.
        targets: List of ground truth textlines.
        threshold: Threshold for IoU.

    Returns:
        precision, recall, F1 score
    """
    intersects = []
    unions = []

    for pred, tar in product(predictions, targets):
        intersects.append(pred.intersection(tar.buffer(0)).area)
        unions.append(pred.union(tar.buffer(0)).area)

    ious = (torch.tensor(intersects) / torch.tensor(unions)).reshape(len(predictions), len(targets))

    mapping = []
    pred_line_list = list(range(len(predictions)))
    gt_line_list = list(range(len(targets)))
    while True:
        max_index = int(torch.argmax(ious))
        max_row = max_index // ious.size(1)
        max_col = max_index % ious.size(1)
        max_value = ious[max_row, max_col].item()

        if max_value <= 0.0:
            break

        ious[max_row, :] = 0
        ious[:, max_col] = 0

        mapping.append((max_row, max_col))
        pred_line_list.remove(max_row)
        gt_line_list.remove(max_col)

    mapping += [(i, -1) for i in pred_line_list]
    mapping += [(-1, i) for i in gt_line_list]

    return mapping


def evaluation(prediction_file: str, ground_truth_file: str) -> Tuple[List[Tuple[int, int]], float, list, list]:
    """
    Evaluates the baseline detection.

    Args:
        prediction_file: Path to the prediction file.
        ground_truth_file: Path to the ground truth file.

    Returns:

    """
    pred_polygone, pred_texts = extract_textlines(prediction_file)
    gt_polygone, gt_texts = extract_textlines(ground_truth_file)

    pred_polygons = [Polygon(poly) for poly in pred_polygone]
    truth_polygons = [Polygon(poly) for poly in gt_polygone]

    # mapping lines bases on iou
    # lines without a match are mapped with the empty string
    mapping = matching(pred_polygons, truth_polygons)
    pred_texts += ['']
    gt_texts += ['']

    pred_texts = [pred_texts[i] for i, _ in mapping]
    gt_texts = [gt_texts[j] for _, j in mapping]

    results = levensthein_distance(gt=[gt_texts],
                                   ocr=[pred_texts],
                                   confidence_list=[[1] * len(pred_texts)])
    lev_dis, lev_med, ratio_list, distance_list, _, bleu_score = results

    char_ratio = calculate_ratio(distance_list)

    correct_lines = np.array(ratio_list)[np.array(ratio_list) == 1.0].tolist()
    bad_lines = np.array(ratio_list)[np.array(ratio_list) < 0.9].tolist()
    print(f"{prediction_file} correct lines: "
          f"{len(correct_lines) / len(ratio_list)}")
    print(
        f"{prediction_file} bad lines: {len(bad_lines) / len(ratio_list)}")
    print(f"{prediction_file} normalized levensthein distance per line: {lev_dis}")
    print(f"{prediction_file} normalized levensthein distance per character: {char_ratio}")
    print(f"{prediction_file} levensthein median: {lev_med}")
    print(f"{prediction_file} levensthein worst line: {min(ratio_list)}\n")
    print(f"{prediction_file} bleu score normalized per line: {bleu_score}")

    return distance_list, bleu_score, correct_lines, bad_lines


# pylint: disable=duplicate-code
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
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=time.time(),
        help="Evaluation name. Results will be printed in 'results_name.json'"
    )

    return parser.parse_args()


# pylint: disable=duplicate-code
def main() -> None:
    """Evaluates baseline predicts."""
    args = get_args()

    pred_dir = adjust_path(args.prediction_dir)
    target_dir = adjust_path(args.ground_truth_dir)

    targets = list(glob.glob(f"{target_dir}/*.xml"))
    predictions = [f"{pred_dir}/{os.path.basename(x)}" for x in targets]

    overall_distance_list = []
    overall_correct_lines = []
    overall_bad_lines = []
    ratios = []
    bleu_sum = 0.0

    for prediction, target in tqdm(zip(predictions, targets), total=len(targets),
                                   desc='evaluation'):
        distance_list, bleu_score, correct_lines, bad_lines = evaluation(prediction, target)
        bleu_sum += bleu_score
        overall_distance_list += distance_list
        overall_correct_lines += correct_lines
        overall_bad_lines += bad_lines
        ratios.append(calculate_ratio(distance_list))

    if len(targets) > 0:
        overall_ratio = calculate_ratio(overall_distance_list)
        print(args.name)
        print(f"\n\noverall levensthein distance per character: {overall_ratio}")
        # print(
        #     f"levensthein distance per character per page: {np.mean(ratios)}
        #     ({np.median(ratios)}) +- {np.std(ratios)} "f"min:{np.min(ratios)} max: "
        #     f"{np.max(ratios)}")
        print(f"Bleu score normalized per line and page: {bleu_sum / len(targets)}")
        print(f"overall correct lines: "
              f"{len(overall_correct_lines) / len(overall_distance_list)}")
        print(
            f"overall bad lines: {len(overall_bad_lines) / len(overall_distance_list)}")

        with open(f'results_{args.name}.json', 'w', encoding='utf8') as json_file:
            json.dump({"levenshtein": overall_ratio, "correct":
                len(overall_correct_lines) / len(overall_distance_list),
                       "bad": len(overall_bad_lines) / len(overall_distance_list)}, json_file)


if __name__ == '__main__':
    main()
