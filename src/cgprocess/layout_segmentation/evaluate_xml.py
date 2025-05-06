# type: ignore
# todo: make this mypy typechecking conform

"""Module for evaluating predictions from xml data, instead of in the training environment."""
import argparse
import glob
import os
from itertools import product
from typing import Dict, List, Optional

import numpy as np
import torch
from bs4 import BeautifulSoup
from shapely.geometry import Polygon
from shapely.validation import explain_validity
from tabulate import tabulate
from tqdm import tqdm


def read_xml(xml_path: str) -> Dict[str:list]:
    """Read data from xml file, returning dictionary with keys for all classes."""
    data = {
        "caption": [],
        "table": [],
        "paragraph": [],
        "heading": [],
        "header": [],
        "separator_vertical": [],
        "separator_horizontal": [],
        "image": [],
        "inverted_text": [],
    }

    # Read the XML file content
    with open(xml_path, "r", encoding="utf-8") as file:
        xml_content = file.read()

    # Parse the XML content with BeautifulSoup
    soup = BeautifulSoup(xml_content, "xml")

    # Iterate over all TextRegion elements
    for text_region in soup.find_all(
        ["TextRegion", "TableRegion", "GraphicRegion", "SeparatorRegion"]
    ):
        # Extract the custom attribute
        custom_attr = text_region.get("custom", "")

        # Extract the type from the custom attribute
        if "structure {type:" in custom_attr:
            type_start = custom_attr.index("structure {type:") + len("structure {type:")
            type_end = custom_attr.index(";", type_start)
            type_value = custom_attr[type_start:type_end]

            # Check if this type is in the list of desired types
            if type_value in data:
                # Extract Coords element and get the points
                coords = text_region.find("Coords")
                if coords:
                    points_str = coords.get("points", "")
                    # Convert points string to a list of tuples
                    points = [
                        tuple(map(int, point.split(",")))
                        for point in points_str.split()
                    ]
                    # Create a Shapely Polygon and add to the list
                    polygon = Polygon(points)
                    data[type_value].append(polygon.buffer(0))

    return data


def remove_duplicate_points(polygon: Polygon) -> Polygon:
    """Remove non unique points from polygon."""
    coords = list(polygon.exterior.coords)
    unique_coords = []
    for coord in coords:
        if coord not in unique_coords:
            unique_coords.append(coord)
    return Polygon(unique_coords)


def calc_metrics(ious: torch.Tensor, threshold: float = 0.7) -> Dict[str, float]:
    """
    calculates precision, recall and f1_score for iou matrix.

    Args:
        ious: matrix with   over union of textline polygons

    Returns:
        precision, recall and f1_score
    """
    tp = 0
    while (torch.tensor(ious.shape) > 0).all():
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
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1_score = (
        0
        if precision + recall == 0
        else 2 * (precision * recall) / (precision + recall)
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "count": ious.shape[1],
    }


def detection_metrics(
    prediction: List[Polygon],
    target: List[Polygon],
    threshold: float = 0.5,
    pred_classes: Optional[List[str]] = None,
    gt_classes: Optional[List[str]] = None,
) -> Dict[str, float]:
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

    for pred_idx, tar_idx in product(range(len(prediction)), range(len(target))):
        pred = prediction[pred_idx]
        tar = target[tar_idx]
        if (
            pred_classes is None
            or gt_classes is None
            or pred_classes[pred_idx] == gt_classes[tar_idx]
        ):
            if pred.is_valid and tar.is_valid:
                intersects.append(pred.intersection(tar).area)
                unions.append(pred.union(tar).area)
            else:
                intersects.append(0.0)
                unions.append(1.0)
                print(f"polygone not valid! {pred.is_valid=} and {tar.is_valid=}")
                print(explain_validity(tar))
        else:
            intersects.append(0.0)
            unions.append(1.0)

    ious = (torch.tensor(intersects) / torch.tensor(unions)).reshape(
        len(prediction), len(target)
    )

    results = calc_metrics(ious, threshold=threshold)

    return results


def compare(pred_xml: str, gt_xml: str, threshold: float = 0.5) -> dict:
    """Compare prediction and ground truth based on classes."""
    categories = [
        "caption",
        "table",
        "paragraph",
        "heading",
        "header",
        "separator_vertical",
        "separator_horizontal",
        "image",
        "inverted_text",
    ]

    data = {key: {} for key in categories}

    pred_objects = read_xml(pred_xml)
    gt_objects = read_xml(gt_xml)

    for cat in categories:
        if pred_objects[cat] or gt_objects[cat]:
            data[cat] = detection_metrics(pred_objects[cat], gt_objects[cat])

    pred_all = [item for sublist in pred_objects.values() for item in sublist]
    gt_all = [item for sublist in gt_objects.values() for item in sublist]

    pred_classes = [key for key, values in pred_objects.items() for _ in values]
    gt_classes = [key for key, values in gt_objects.items() for _ in values]

    data["all"] = detection_metrics(
        pred_all,
        gt_all,
        threshold=threshold,
        pred_classes=pred_classes,
        gt_classes=gt_classes,
    )

    return data


def get_args() -> argparse.Namespace:
    # pylint: disable=duplicate-code
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="evaluate_xml")

    parser.add_argument(
        "--pred_dir",
        "-p",
        type=str,
        default=None,
        help="path for folder with images. Images need to be jpg.",
    )

    parser.add_argument(
        "--gt_dir",
        "-g",
        type=str,
        default=None,
        help="path for folder with layout xml files.",
    )

    return parser.parse_args()


def print_table(count, tp, fp, fn, precision, recall, f1_score):
    """Assembles table in md format for layout evaluation results."""
    categories = [
        "caption",
        "table",
        "paragraph",
        "heading",
        "header",
        "separator_vertical",
        "separator_horizontal",
        "image",
        "inverted_text",
        "all",
    ]

    # Prepare data for tabulate
    table_data = []

    for cat in categories:
        # Calculate metrics for table
        precision_mean = np.mean(precision[cat])
        precision_median = np.median(precision[cat])
        precision_std = np.std(precision[cat])

        recall_mean = np.mean(recall[cat])
        recall_median = np.median(recall[cat])
        recall_std = np.std(recall[cat])

        f1_mean = np.mean(f1_score[cat])
        f1_median = np.median(f1_score[cat])
        f1_std = np.std(f1_score[cat])

        precision_score = np.sum(tp[cat]) / (np.sum(tp[cat]) + np.sum(fp[cat]))
        recall_score = np.sum(tp[cat]) / (np.sum(tp[cat]) + np.sum(fn[cat]))
        f1_score_calc = (
            2 * (precision_score * recall_score) / (precision_score + recall_score)
        )

        # Append row to table data
        table_data.append(
            [
                cat,
                f"{count[cat]}",
                f"{precision_mean:.4f} ({precision_median:.4f}) ± {precision_std:.4f}",
                f"{recall_mean:.4f} ({recall_median:.4f}) ± {recall_std:.4f}",
                f"{f1_mean:.4f} ({f1_median:.4f}) ± {f1_std:.4f}",
                f"{precision_score:.4f}",
                f"{recall_score:.4f}",
                f"{f1_score_calc:.4f}",
            ]
        )

    # Define headers for the table
    headers = [
        "Category",
        "count in ground truth",
        "Precision (mean(median) ± std)",
        "Recall (mean(median) ± std)",
        "F1 Score (mean(median) ± std)",
        "Precision (overall)",
        "Recall (overall)",
        "F1 Score (overall)",
    ]

    # Use tabulate to format the data into a table and print
    print(tabulate(table_data, headers, tablefmt="grid"))


def main():
    """Assemble data and print table to command line."""
    args = get_args()

    categories = [
        "caption",
        "table",
        "paragraph",
        "heading",
        "header",
        "separator_vertical",
        "separator_horizontal",
        "image",
        "inverted_text",
        "all",
    ]

    pred_paths = list(glob.glob(f"{args.pred_dir}/*.xml"))
    gt_paths = [f"{args.gt_dir}/{os.path.basename(x)}" for x in pred_paths]

    # pred_paths = ['data/pipeline_test_dataset/Koelnische_Zeitung_1866-06_1866-09_Anzeigen_0308.xml']
    # gt_paths=['data/newSplit2/test/annos/Koelnische_Zeitung_1866-06_1866-09_Anzeigen_0308.xml']

    tp = {c: [] for c in categories}
    fp = {c: [] for c in categories}
    fn = {c: [] for c in categories}
    precision = {c: [] for c in categories}
    recall = {c: [] for c in categories}
    f1_score = {c: [] for c in categories}
    count = {c: 0 for c in categories}

    for pred, gt in tqdm(zip(pred_paths, gt_paths), total=len(pred_paths)):
        data = compare(pred, gt)
        for cat in categories:
            if data[cat]:
                count[cat] += data[cat]["count"]
                tp[cat].append(data[cat]["TP"])
                fp[cat].append(data[cat]["FP"])
                fn[cat].append(data[cat]["FN"])
                precision[cat].append(data[cat]["precision"])
                recall[cat].append(data[cat]["recall"])
                f1_score[cat].append(data[cat]["f1"])

    print_table(count, tp, fp, fn, precision, recall, f1_score)


if __name__ == "__main__":
    main()
