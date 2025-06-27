"""Evaluation script for yolo."""
import argparse
import glob
import json
import os
import re
import warnings
from os.path import basename
from typing import Tuple, List

import numpy as np
import torch
from bs4 import BeautifulSoup, PageElement
from shapely.geometry import Polygon
from skimage.draw import polygon as sk_polygon
from torchmetrics.classification import MulticlassConfusionMatrix
from tqdm import tqdm

from cgprocess.layout_segmentation.helper.train_helper import multi_precison_recall
from cgprocess.layout_segmentation.utils import adjust_path

# Labels to evaluate
EVAL_LABELS = ["paragraph", "inverted_text", "table", "caption", "heading", "header", "image"]

MAPPING = {'article': 'paragraph',
           'inverted_text': 'inverted_text',
           'paragraph': 'paragraph',
           'caption': 'caption',
           'heading': 'heading',
           'header': 'header',
           'Image': 'image',
           'image': 'image',
           'newspaper_header': 'header',
           'headline': 'heading',
           'table': 'table'}


def get_tag(textregion: PageElement) -> str:
    """
    Returns the tag of the given textregion.

    Args:
        textregion: PageElement of Textregion

    Returns:
        Given tag of that Textregion
    """
    desc = textregion['custom']
    match = re.search(r"\{type:.*;\}", desc)
    if match is None:
        return 'UnknownRegion'

    tag = match.group()[6:-2]
    if tag == 'article_':
        tag = "paragraph"
    if tag == 'article':
        tag = "paragraph"
    return tag


def multi_class_f1(
        pred: torch.Tensor, target: torch.Tensor, metric: MulticlassConfusionMatrix
) -> torch.Tensor:
    """
    Calculates the f1 score.

    Calculate csi score using true positives, true negatives and false negatives from confusion matrix.
    Csi score is used as substitute for accuracy, calculated separately for each class.
    Returns numpy array with an entry for every class. If every prediction is a true negative,
    the score can't be calculated and the array will contain nan. These cases should be completely ignored.

    Args:
        pred: prediction tensor
        target: target tensor
    """
    pred = pred.flatten()
    target = target.flatten()

    matrix: torch.Tensor = metric(pred, target)
    true_positive = torch.diagonal(matrix)
    false_positive = torch.sum(matrix, dim=1) - true_positive
    false_negative = torch.sum(matrix, dim=0) - true_positive
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csi = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    return csi


def read_json(file_path: str):
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    bboxes = data.get('bboxes', [])

    region_coords = []
    region_labels = []

    for bbox_data in bboxes:
        label = bbox_data.get("class", "No Label")
        label = MAPPING.get(label, 'NoLabel')

        if label not in EVAL_LABELS:
            continue

        region_labels.append(label)

        bbox = bbox_data.get('bbox', {})

        # Convert bbox to a 2D tensor representing the polygon
        polygon = Polygon([
            [bbox['x0'], bbox['y0']],  # Top-left corner
            [bbox['x1'], bbox['y0']],  # Top-right corner
            [bbox['x1'], bbox['y1']],  # Bottom-right corner
            [bbox['x0'], bbox['y1']]  # Bottom-left corner
        ])

        region_coords.append(polygon)

    return region_coords, region_labels


def read_xml(file_path: str) -> Tuple[List[Polygon], List[str], int, int]:
    """
    Reads in the given XML file.

    Args:
        file_path: path to the XML file.

    Returns:

    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    width, height = int(page["imageWidth"]), int(page["imageHeight"])
    region_coords = []
    region_labels = []

    text_regions = page.find_all(['TextRegion', 'GraphicRegion'])
    for region in text_regions:
        coords = region.find_all('Coords')[0]
        label = MAPPING.get(get_tag(region), "NoLabel")
        polygon = torch.tensor([tuple(map(int, point.split(','))) for
                                point in coords['points'].split()])

        if len(polygon) <= 3:
            print(f"Found invalid TextRegion in {file_path}.")
            continue

        if label not in EVAL_LABELS:
            continue

        region_labels.append(label)
        region_coords.append(polygon)

    table_regions = page.find_all('TableRegion')
    for region in table_regions:
        coords = region.find_all('Coords')[0]
        polygon = torch.tensor([tuple(map(int, point.split(','))) for
                                point in coords['points'].split()])
        if len(polygon) <= 3:
            print(f"Found invalid table in {file_path}.")
            continue

        region_labels.append('table')
        region_coords.append(polygon)

    return [Polygon(poly) for poly in region_coords], region_labels, height, width


def sort_polygons_and_labels(polygons: List[Polygon], labels: List[str]):
    """
    Sorts polygons and labels based on a given order of labels.

    Args:
        polygons: List of polygons.
        labels: List of labels corresponding to the polygons.

    Returns:
        A tuple containing two lists: sorted polygons and sorted labels.
    """
    # if lists are empty return
    if len(polygons) == 0:
        return [], []

    # Create a dictionary to map each label to its index in the order list
    order_index = {label: i for i, label in enumerate(EVAL_LABELS)}

    # Create a list of tuples containing (label, polygon) and sort it based on label's order index
    sorted_pairs = sorted(zip(labels, polygons),
                          key=lambda pair: order_index.get(pair[0], float('inf')))

    # Unzip the sorted pairs into two lists: sorted_labels and sorted_polygons
    sorted_labels, sorted_polygons = zip(*sorted_pairs)

    # Convert sorted_labels and sorted_polygons back to lists and return
    return list(sorted_polygons), list(sorted_labels)


def draw_image(polygons: List[Polygon], labels: List[str], shape: Tuple[int, int]) -> torch.Tensor:
    """
    Draws image using the given polygons and labels.

    Args:
        polygons: List of polygons.
        labels: List of labels corresponding to the polygons.
        shape: Shape of the image.

    Returns:
        numpy array containing the image.
    """
    # Create an empty numpy array
    arr = torch.zeros(shape, dtype=torch.uint8)

    polygons, labels = sort_polygons_and_labels(polygons, labels)

    # Iterate through each polygon
    for polygon, label in zip(polygons, labels):
        # Get the exterior coordinates
        exterior_coords = torch.tensor(polygon.exterior.coords, dtype=torch.int32)

        # Use skimage.draw.polygon to fill the polygon in the array
        rr, cc = sk_polygon(exterior_coords[:, 1], exterior_coords[:, 0], shape)
        arr[rr, cc] = EVAL_LABELS.index(label) + 1

    return arr


def evaluate(target: str, prediction: str):
    """
    Calculates the f1 score the given prediction and traget.

    Args:
        target: Path to target xml file.
        prediction: Path to prediction xml file.

    Returns:

    """
    pred_polygons, pred_labels = read_json(prediction)
    tar_polygons, tar_labels, width, height = read_xml(target)

    pred_tensor = draw_image(pred_polygons, pred_labels, shape=(width, height)).flatten()
    tar_tensor = draw_image(tar_polygons, tar_labels, shape=(width, height)).flatten()

    if 'inverted_text' in pred_labels:
        print("inverted_text in predictions")

    if 'inverted_text' in tar_labels:
        print("inverted_text in targets")

    _, _, f1_score, pixel_counts = multi_precison_recall(pred_tensor, tar_tensor)

    return f1_score.numpy(), pixel_counts.numpy()


def main():
    """
    Calculates the f1 score the given prediction and target sets.
    """
    args = get_args()

    pred_dir = adjust_path(args.prediction_dir)
    target_dir = adjust_path(args.ground_truth_dir)

    if args.custom_split_file:
        with open(args.custom_split_file, "r", encoding="utf-8") as file:
            split = json.load(file)
            targets = [f"{target_dir}/{x}.xml" for x in split[args.split]]
    else:
        targets = list(glob.glob(f"{target_dir}/*.xml"))

    predictions = [f"{pred_dir}/{basename(x)[:-4]}.json" for x in targets]

    class_f1_list = []
    class_f1_weights = []
    for target, prediction in tqdm(zip(targets, predictions), total=len(targets),
                                   desc="Evaluating"):
        f1_values, size = evaluate(target, prediction)
        class_f1_list.append(np.nan_to_num(f1_values, nan=0))
        class_f1_weights.append(size)

    # Convert lists to arrays
    class_f1_array = np.array(class_f1_list)  # shape: (num_samples, num_classes)
    class_weight_array = np.array(class_f1_weights)  # shape: (num_samples, num_classes)

    mask = np.sum(class_f1_weights, axis=0) == 0
    class_weight_array[:, mask] = 1

    batch_class_f1 = np.average(class_f1_array, axis=0, weights=class_weight_array)

    batch_class_f1[mask] = np.nan

    os.makedirs(f"{args.prediction_dir}/evaluation", exist_ok=True)
    if not os.path.exists(f"{args.prediction_dir}/evaluation/evaluation.txt"):
        with open(f"{args.prediction_dir}/evaluation/evaluation.txt", 'w', encoding="utf-8") as file:
            file.write("")

    with open(f"{args.prediction_dir}/evaluation/evaluation.txt", "a", encoding="utf-8") as file:
        file.writelines([
            "\n",
            f"{args.split}: \n",
            f"{args.prediction_dir=}: \n",
            f"{args.ground_truth_dir=}: \n",
            "f1 scores: \n",
        ])
        file.writelines([f"{label}: {batch_class_f1[idx]}\n" for idx, label in
                         enumerate(["background"] + EVAL_LABELS)])
        file.writelines(["\n", "\n", ])


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
        "--custom-split-file",
        type=str,
        default=None,
        help="Provide path for custom split json file. This should contain a list with file stems "
             "of train, validation and test images. This will only evaluate the test dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Test",
        help="Choose the key from the split-file, default is 'Test'",
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
