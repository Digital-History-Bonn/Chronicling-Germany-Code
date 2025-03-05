"""Merge npz or json files that have been generated from xml data."""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def merge_files(args, output_path, paths) -> None:
    """Merge data and save it to disc."""
    page_csv = []
    page_json = []
    for path in tqdm(paths):
        if args.csv:
            region_csv = np.load(output_path / "temp" / f"{path}.npz")['array']
            page_csv.append(region_csv)
        if args.json:
            with open(output_path / "temp" / f"{path}.npz", "r", encoding="utf-8") as file:
                region_json = json.load(file)
            page_json.append({"path": path, "regions": region_json})
    if args.csv:
        csv_data = np.vstack(page_csv).tolist()
        with open(output_path / "text_export.csv", 'w', newline='', encoding='utf-8') as file:
            data_writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            if args.lines:
                data_writer.writerow(["path", "region", "line", "class", "confidence", "text"])
            else:
                data_writer.writerow(["path", "region", "class", "confidence", "text"])
            for row in csv_data:
                data_writer.writerow(row)
    if args.json:
        with open(output_path / "text_export.json", 'w', newline='', encoding='utf-8') as file:
            json.dump(page_json, file)
    shutil.rmtree(output_path / "temp", ignore_errors=True)


def main(args: argparse.Namespace) -> None:
    """
    Load xml files and assemble page lists before saving them.
    """
    output_path = Path(args.output_path)

    # todo: add json handling
    paths = [
        f[:-4] for f in os.listdir(output_path / "temp") if f.endswith(".npz")
    ]

    merge_files(args, output_path, paths)


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        dest="output_path",
        default="data/output-text/",
        help="path for output folder",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Activates csv export.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Activates json export.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parameter_args = get_args()
    assert parameter_args.csv or parameter_args.json, ("Please activate at least one export methon with '--csv' or "
                                                       "'--json'.")
    main(parameter_args)
