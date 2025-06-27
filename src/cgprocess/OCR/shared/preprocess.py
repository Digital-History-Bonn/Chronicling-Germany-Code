"""Splits and preprocesses data for training."""

import argparse
import json
import os
from typing import List

from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm

from cgprocess.OCR.shared.utils import adjust_path, pad_image, pad_xml


def copy_and_pad_xml(input_path: str, output_dir: str, pad_value: int = 10) -> None:
    """
    Pads all points in the XML file and saves the modified XML file.

    Args:
        input_path: Path to the input XML file.
        output_dir: Directory where the modified XML file will be saved.
        pad_value: The padding value (int) to apply to all points.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the XML file
    with open(input_path, "r", encoding="utf-8") as file:
        xml_content = file.read()

    # Parse the XML content
    soup = BeautifulSoup(xml_content, "xml")
    soup = pad_xml(soup, pad_value=pad_value)

    # Extract the file name from the input path
    file_name = os.path.basename(input_path)

    # Create the full output path
    output_path = os.path.join(output_dir, file_name)

    # Save the modified XML content
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(str(soup))


def copy_and_pad_image(input_path: str, output_dir: str, pad_value: int = 10) -> None:
    """
    Loads an image, pads it on all sides and saves it at the specified location.

    Args:
        input_path: Path to the input image file.
        output_dir: Path where the padded image will be saved.
        pad_value: The padding value (int) to apply on all sides of the image.
    """
    # Load the image
    image = Image.open(input_path)

    padded_image = pad_image(image, pad=pad_value)

    # Save the padded image
    file_name = os.path.basename(input_path)
    padded_image.save(os.path.join(output_dir, file_name))


def preprocess(
    train_paths: List[str],
    valid_paths: List[str],
    test_paths: List[str],
    image_dir: str,
    annotation_dir: str,
    output_dir: str,
) -> None:
    """
    Preprocesses and splits data and saves it.

    Args:
        train_paths: Paths to the training images.
        valid_paths: Paths to the validation images.
        test_paths: Paths to the test images.
        image_dir: Path to the image folder.
        annotation_dir: Path to the annotation folder.
        output_dir: Path to output dictionary.
    """
    # Create directories if they don't exist
    train_dir = f"{output_dir}/train"
    valid_dir = f"{output_dir}/valid"
    test_dir = f"{output_dir}/test"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy file pairs to the respective directories
    for base_path in tqdm(train_paths, desc="Preprocessing Training data"):
        copy_and_pad_image(f"{image_dir}/{base_path}.jpg", train_dir)
        copy_and_pad_xml(f"{annotation_dir}/{base_path}.xml", train_dir)

    for base_path in tqdm(valid_paths, desc="Preprocessing Validation data"):
        copy_and_pad_image(f"{image_dir}/{base_path}.jpg", valid_dir)
        copy_and_pad_xml(f"{annotation_dir}/{base_path}.xml", valid_dir)

    for base_path in tqdm(test_paths, desc="Preprocessing Test data"):
        copy_and_pad_image(f"{image_dir}/{base_path}.jpg", test_dir)
        copy_and_pad_xml(f"{annotation_dir}/{base_path}.xml", test_dir)


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    # pylint: disable=duplicate-code
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument(
        "--images",
        "-i",
        type=str,
        default=None,
        help="path for folder with images. Need to be jpg."
    )

    parser.add_argument(
        "--annotations",
        "-a",
        type=str,
        default=None,
        help="path for folder with annotation xml files."
    )

    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default=None,
        help="path to the split json."
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="path to the folder where to save the preprocessed files",
    )

    return parser.parse_args()


def main() -> None:
    """Starts the preprocessing process."""
    args = get_args()

    images_dir = adjust_path(args.images)
    annotation_dir = adjust_path(args.annotations)
    output_dir = adjust_path(args.output)

    # check args
    if images_dir is None:
        raise ValueError("Please enter a valid path to image data!")

    if annotation_dir is None:
        raise ValueError("Please enter a valid path to annotation data!")

    if output_dir is None:
        raise ValueError("Please enter a valid output path!")

    # pylint: disable=duplicate-code
    with open(args.split, "r",
              encoding="utf-8") as file:
        data = json.load(file)

    train_files = data.get("Training", [])
    valid_files = data.get("Validation", [])
    test_files = data.get("Test", [])

    print(f"{len(train_files)=}")
    print(f"{len(valid_files)=}")
    print(f"{len(test_files)=}")

    preprocess(
        train_files, valid_files, test_files, images_dir, annotation_dir, output_dir
    )


if __name__ == "__main__":
    main()
