"""Script for downloading dataset and our models"""
import argparse
import os
import zipfile
from typing import Optional

import requests

DATASET_URL = ("https://gitlab.uni-bonn.de/digital-history/Chronicling-Germany-Dataset/-/"
               "archive/main/Chronicling-Germany-Dataset-main.zip?path=data")
MODELS_URL = ("https://gitlab.uni-bonn.de/digital-history/Chronicling-Germany-Dataset/-/"
              "archive/main/Chronicling-Germany-Dataset-main.zip?path=models")


def download_extract(url: str, target_path: str):
    """
    Downloads data from given url, saves and extracts it to target folder.

    Args:
        url (str): URL to download.
        target_path (str): Path where to save the extracted files.
    """
    # create folder
    os.makedirs(target_path, exist_ok=True)

    # download
    r = requests.get(url, timeout=10)
    with open(f"{target_path}/data.zip", 'wb') as f:
        f.write(r.content)

    # unzip data
    with zipfile.ZipFile(f"{target_path}/data.zip", 'r') as zipper:
        zipper.extractall(path=target_path)

    os.remove(f"{target_path}/data.zip")


def adjust_path(path: Optional[str]) -> Optional[str]:
    """
    Make sure, there is a slash at the end of a (folder) spath string.

    Args:
        path: String representation of path

    Returns:
        path without ending '/'
    """
    return path if not path or path[-1] != '/' else path[:-1]


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="Download dataset and our models")

    parser.add_argument(
        "--all",
        action="store_true",
        help="downloads dataset and models if set",
    )

    parser.add_argument(
        "--dataset",
        action="store_true",
        help="downloads dataset if set",
    )

    parser.add_argument(
        "--dataset-path",
        "-dp",
        type=str,
        dest="dataset_path",
        default="data",
        help="Path to folder where to save dataset.",
    )

    parser.add_argument(
        "--models",
        action="store_true",
        help="downloads models if set",
    )

    parser.add_argument(
        "--model-path",
        "-mp",
        type=str,
        dest="model_path",
        default="models",
        help="Path to folder where to save the models.",
    )

    return parser.parse_args()


def main():
    """Downloads dataset and our models."""
    print('Please be patient downloading process can take a while...')
    args = get_args()

    dataset_path = adjust_path(args.dataset_path)
    model_path = adjust_path(args.model_path)

    if args.dataset or args.all:
        print("downloading dataset ...")
        download_extract(DATASET_URL, dataset_path)

    if args.models or args.all:
        print("downloading models ...")
        download_extract(MODELS_URL, model_path)


if __name__ == '__main__':
    main()
