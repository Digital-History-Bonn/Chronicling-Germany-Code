"""module for validating dataset. Can be used to testwise load the entire Dataset"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.cgprocess.layout_segmentation.utils import draw_prediction


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="validate dataset")
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        dest="data_path",
        default=None,
        help="path for folder with folders 'images' and 'targets'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parameter_args = get_args()
    data_path = Path(parameter_args.data_path)
    for path in tqdm(data_path.iterdir(), desc="drawing targets", unit="files"):
        if path.suffix == ".npz":
            img = np.load(path)["array"]
            draw_prediction(img, data_path / (path.stem + ".png"))
