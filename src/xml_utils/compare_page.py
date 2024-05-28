"""Module for comparing the transcript of entire pages."""

import argparse
import json
import os

import Levenshtein
from bs4 import BeautifulSoup
from tqdm import tqdm
# from nltk.translate.bleu_score import corpus_bleu

from src.xml_utils.read_xml import line_has_text


def extract_text(xml_data: BeautifulSoup, split: bool = True):
    """
    For actual TextRegions this function assigns uncertain texts on each line, that has a big difference to the
    verify data, or contains a lot of numbers and special characters. Regions, that should not be Text Regions are
    being renamed."""
    text_regions = xml_data.find_all("TextRegion")

    lines_text = []

    for region in text_regions:
        lines = region.find_all("TextLine")
        for line in lines:
            if line_has_text(line):
                lines_text.append(
                    line.TextEquiv.Unicode.contents[0].split(" ") if split else line.TextEquiv.Unicode.contents[0])
    return lines_text


def main(args: argparse.Namespace):
    """Main function for compare pages, loads xml data and compares extracted text."""
    if not os.path.exists(args.output_path):
        print(f"creating {args.output_path}.")
        os.makedirs(args.output_path)

    paths = [
        f[:-4] for f in os.listdir(args.ocr_path) if f.endswith(".xml")
    ]

    if args.custom_split_file:
        with open(args.custom_split_file, "r", encoding="utf-8") as file:
            split = json.load(file)
            paths = split[2]

    result_list = []
    for path in tqdm(paths):
        print(path)
        with open(f"{args.ocr_path}{path}.xml", "r", encoding="utf-8") as file:
            data = file.read()

        ocr_data = BeautifulSoup(data, "xml")
        ocr_text = extract_text(ocr_data, split=False)
        with open(f"{args.gt_path}{path}.xml", "r", encoding="utf-8") as file:
            gt_data = file.read()
        gt_data = BeautifulSoup(gt_data, "xml")
        gt_text = extract_text(gt_data, split=False)

        # result_list.append(corpus_bleu([ocr_text] * len(gt_text), gt_text))
        result_list.append(Levenshtein.distance("\n".join(ocr_text), "\n".join(gt_text))/(len(gt_text) + len(ocr_text)))
    print(sum(result_list) / len(result_list))


# pylint: disable=duplicate-code
def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument(
        "--ocr-path",
        "-oc",
        type=str,
        default="data/ocr/",
        help="path for ocr folder",
    )
    parser.add_argument(
        "--gt-path",
        "-gt",
        type=str,
        default="data/ground_truth",
        help="folder path for ground truth folder"
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        dest="output_path",
        default="data/output-page/",
        help="path for output folder",
    )
    parser.add_argument(
        "--custom-split-file",
        type=str,
        default=None,
        help="Provide path for custom split json file. This should contain a list with file stems "
             "of train, validation and test images. This will only evaluate the test dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parameter_args = get_args()
    main(parameter_args)
