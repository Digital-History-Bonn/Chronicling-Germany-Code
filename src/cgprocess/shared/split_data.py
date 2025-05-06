"""Script for splitting the dataset"""

import argparse
import json
import os
import shutil

from tqdm import tqdm

from src.cgprocess.layout_segmentation.utils import adjust_path


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    # pylint: disable=duplicate-code
    parser = argparse.ArgumentParser(description="split dataset")

    parser.add_argument(
        "--custom-split-file",
        type=str,
        default=None,
        help="Provide path for custom split json file. This should contain a list with file stems "
        "of train, validation and test images. File stem is the file name without the extension.",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Extract specified datasets. Dataset name has to match the name "
        "inside the file specified through --custom-split-file. "
        "Multiple dataset names can be supplied.",
    )

    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        dest="data_path",
        default=None,
        help="path for folder with folders 'images' and 'annotations'",
    )

    return parser.parse_args()


def main() -> None:
    """
    Copies images and xml files according to the specified split. As target a 'datasets' folder is created, in which
    one folder for each dataset is created. xml files will be copied into a folder named page inside the respective
    dataset folder.
    """
    args = get_args()

    dataset_names = args.datasets
    data_path = adjust_path(args.data_path)
    image_path = f"{data_path}images/"
    annotations_path = f"{data_path}annotations/"

    if not os.path.exists(f"{data_path}datasets/"):
        os.makedirs(f"{data_path}datasets/")

    with open(args.custom_split_file, "r", encoding="utf-8") as file:
        split = json.load(file)
        for dataset_name in dataset_names:
            dataset = split[dataset_name]

            dataset_path = f"{data_path}datasets/{dataset_name}/"
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            if not os.path.exists(dataset_path + "page/"):
                os.makedirs(dataset_path + "page/")

            for path in tqdm(dataset, desc=f"Copy {dataset_name} data"):
                shutil.copy2(image_path + path + ".jpg", dataset_path + path + ".jpg")
                shutil.copy2(
                    annotations_path + path + ".xml",
                    dataset_path + "page/" + path + ".xml",
                )


if __name__ == "__main__":
    main()
