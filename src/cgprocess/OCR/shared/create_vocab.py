"""Extract a vocabulary containing all unique characters from the XML data."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict

from bs4 import BeautifulSoup, ResultSet
from tqdm import tqdm

from cgprocess.OCR.shared.utils import line_has_text


def create_vocab(regions: ResultSet, vocab: Dict[str, int]) -> None:
    """adds characters to vocabulary"""
    for region in regions:
        lines = region.find_all("TextLine")
        for line in lines:
            if line_has_text(line):
                text = line.TextEquiv.Unicode.contents[0]
                for char in text:
                    vocab[char] = vocab.get(char, 0) + 1


def main(parsed_args: argparse.Namespace) -> None:
    """Main method loading xml data and saving extracted vocab."""
    data_path = Path(parsed_args.data_path)
    paths = [f[:-4] for f in os.listdir(data_path) if f.endswith(".xml")]

    vocab: Dict[str, int] = {}

    for path in tqdm(paths):
        with open(data_path / f"{path}.xml", "r", encoding="utf-8") as file:
            data = file.read()

        xml_data = BeautifulSoup(data, "xml")

        text_regions = xml_data.find_all("TextRegion")

        create_vocab(text_regions, vocab)

    with open(data_path / "vocabulary.json", "w", encoding="utf-8") as file:
        json.dump(list(vocab.keys()), file)


def get_args() -> argparse.Namespace:
    # pylint: disable=duplicate-code
    """defines arguments"""
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default="data/transkribus_processing/",
        help="path for folder with xml files from which a tag should be removed",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parameter_args = get_args()
    main(parameter_args)
