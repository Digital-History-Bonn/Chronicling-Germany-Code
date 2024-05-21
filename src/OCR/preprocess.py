"""Splits and preprocesses data for training."""

import glob
import os
from typing import List

import numpy as np
from PIL import Image
from bs4 import BeautifulSoup

from utils import pad_xml, pad_image


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
    with open(input_path, 'r', encoding='utf-8') as file:
        xml_content = file.read()

    # Parse the XML content
    soup = BeautifulSoup(xml_content, 'xml')
    soup = pad_xml(soup, pad_value=pad_value)

    # Extract the file name from the input path
    file_name = os.path.basename(input_path)

    # Create the full output path
    output_path = os.path.join(output_dir, file_name)

    # Save the modified XML content
    with open(output_path, 'w', encoding='utf-8') as file:
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


def preprocess(train_paths: List[str], valid_paths: List[str], test_paths: List[str]) -> None:
    """
    Preprocesses and splits data and saves it.

    Args:
        train_paths: Paths to the training images.
        valid_paths: Paths to the validation images.
        test_paths: Paths to the test images.
    """
    # Create directories if they don't exist
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/valid', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)

    # Copy file pairs to the respective directories
    for base_path in train_paths:
        copy_and_pad_image(f'{base_path}.jpg', 'data/train')
        copy_and_pad_xml(f'{base_path}.xml', 'data/train')

    for base_path in valid_paths:
        copy_and_pad_image(f'{base_path}.jpg', 'data/valid')
        copy_and_pad_xml(f'{base_path}.xml', 'data/valid')

    for base_path in test_paths:
        copy_and_pad_image(f'{base_path}.jpg', 'data/test')
        copy_and_pad_xml(f'{base_path}.xml', 'data/test')


def main():
    """Starts the preprocessing process."""
    np.random.seed(42)
    files = [x[:-4] for x in glob.glob('data/xml_files/*.jpg')]
    order = np.arange(len(files))
    np.random.shuffle(order)
    s1 = int(len(files) * 0.85)
    s2 = int(len(files) * 0.9)

    train_files = [files[i] for i in order[:s1]]
    valid_files = [files[i] for i in order[s1:s2]]
    test_files = [files[i] for i in order[s2:]]

    print(f"{len(train_files)=}")
    print(f"{len(valid_files)=}")
    print(f"{len(test_files)=}")

    preprocess(train_files, valid_files, test_files)


if __name__ == '__main__':
    main()
