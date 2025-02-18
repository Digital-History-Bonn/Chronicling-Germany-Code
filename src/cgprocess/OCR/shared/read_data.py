import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

def merge_data(data_path: Path, paths: List[str]) -> np.ndarray:
    page_list = []
    for path in tqdm(paths):
        page_csv = np.load(data_path / f"{path}.npz")['array']
        page_list.append(page_csv)
    return np.vstack(page_list)

def main(args: argparse.Namespace) -> None:
    """
    Load xml files and assemble page lists before saving them.
    """
    data_path = Path(args.output_path)

    # todo: add json handling
    paths = [
        f[:-4] for f in os.listdir(data_path / "temp") if f.endswith(".npz")
    ]

    data = merge_data(data_path, paths)


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default="data/",
        help="path for data folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parameter_args = get_args()
    main(parameter_args)