"""Dataset class for SSM based OCR training."""

import gzip
import json
import os
import random
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
from PIL import Image
from bs4 import BeautifulSoup
from skimage import draw
from torchvision import transforms
from tqdm import tqdm

from src.cgprocess.OCR.shared.tokenizer import OCRTokenizer
from src.cgprocess.OCR.shared.utils import get_bbox, line_has_text
from src.cgprocess.layout_segmentation.processing.read_xml import xml_polygon_to_polygon_list
from src.cgprocess.shared.datasets import TrainDataset
from src.cgprocess.shared.multiprocessing_handler import run_processes


def preprocess_data(image: torch.Tensor, text_lines: List[BeautifulSoup], image_height: int) -> Tuple[
    List[torch.Tensor], List[str]]:
    texts = []
    crops: List[torch.Tensor] = []
    for line in text_lines:
        if line_has_text(line):
            extract_crop(crops, image, line, image_height)

            texts.append(line.find('Unicode').text)
    return crops, texts


def extract_crop(crops: List[torch.Tensor], image: torch.Tensor, line: BeautifulSoup, image_height: int) -> None:
    """Crops the image according to bbox information and masks all pixels outside the text line polygon.
    Resizes the crop, so that all crops have the same height.
    Args:
        crops:  list to insert all crops
        image: full input image
        line: current xml object with polygon data
        image_height: fixed height for all crops"""
    region_polygon = torch.tensor(xml_polygon_to_polygon_list(line.Coords["points"]))
    bbox = torch.tensor(get_bbox(region_polygon))
    crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    mask = torch.zeros_like(crop)
    mask[draw.polygon(region_polygon[:, 1], region_polygon[:, 0])] = 1
    crop *= mask

    scale = image_height / crop.shape[-2]
    rescale = transforms.Resize((image_height, int(crop.shape[-1] * scale)))
    crops.append(rescale(crop).astype(np.uint8).tolist())


def load_data(image_path: Path, xml_path: Path) -> Tuple[torch.Tensor, List[BeautifulSoup]]:
    """Load image and xml data, transform the PIL image to a torch tensor and extract all xml text line objects."""
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()
    pil_image = Image.open(image_path).convert('L')
    transform = transforms.PILToTensor()
    image = transform(pil_image)
    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    text_lines = page.find_all('TextLine')
    return image, text_lines


def extract_page(queue: Queue, target_paths: list, image_path: Path, target_path: Path, image_extension: str,
                 tokenizer: OCRTokenizer, image_height) -> None:
    """Extract target and input data for OCR SSM training"""
    while True:
        arguments = queue.get()
        if arguments[-1]:
            break
        file_stem, _ = arguments
        if file_stem in target_paths:
            return
        image, text_lines = load_data(image_path / f"{file_stem}{image_extension}",
                                      target_path / f"{file_stem}.xml")
        crops, texts = preprocess_data(image, text_lines, image_height)
        targets = np.array([tokenizer(line).type_as(torch.uint8).tolist() for line in texts])

        json_str = json.dumps({"crops": crops, "texts": texts, "targets": targets})  # 2. string (i.e. JSON)
        json_bytes = json_str.encode('utf-8')  # 3. bytes (i.e. UTF-8)

        with gzip.open(target_path / f"{file_stem}.json", 'w') as file:  # 4. fewer bytes (i.e. gzip)
            file.write(json_bytes)


def get_progress(output_path):
    len([f for f in os.listdir(output_path) if f.endswith(".npz")])


class SSMDataset(TrainDataset):
    """
    Dataset class for SSM based OCR training.
    """

    def __init__(self,
                 kwargs: dict,
                 image_height: int,
                 tokenizer: OCRTokenizer):
        """
        Args:
            image_path: path to folder with images
            target_path: path to folder with xml files
            cfg: configuration file tied to the model
        """
        super().__init__(**kwargs)
        self.image_height = image_height

        self.tokenizer = tokenizer

        self.crops: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.texts: List[str] = []

        self.prepare_data()

    def get_data(self):
        for file_stem in tqdm(self.file_stems, desc="Loading data", unit="Pages"):
            with gzip.open(self.target_path / f"{file_stem}.json", 'r') as file:  # 4. gzip
                json_bytes = file.read()  # 3. bytes (i.e. UTF-8)

            json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
            data = json.loads(json_str)

            self.crops.extend(data["crops"])
            self.targets.extend(data["targets"])
            self.texts.extend(data["texts"])

    def extract_data(self) -> None:
        """Load xml files and save result image.
        Calls read and draw functions"""

        file_stems = [
            f[:-4] for f in os.listdir(self.annotations_path) if f.endswith(".xml")
        ]

        if not self.target_path or not os.path.exists(self.target_path):
            print(f"creating {self.target_path}.")
            os.makedirs(output_path)  # type: ignore

        target_stems = [
            f[:-4] for f in os.listdir(self.target_path) if f.endswith(".npz")
        ]
        path_queue: Queue = Queue()
        total = len(file_stems)

        processes = [Process(target=extract_page,
                             args=(path_queue, target_stems, self.image_path, self.annotations_path, self.image_extension,
                                   self.tokenizer, self.image_height)) for _ in range(self.num_processes)]

        for path in tqdm(file_stems, desc="Put paths in queue"):
            path_queue.put((path, False))

        run_processes({"method": get_progress, "args": self.target_path}, processes, path_queue, total, "Page converting")

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return torch.tensor(self.crops[idx]).float() /255, torch.tensor(self.targets[idx], dtype=torch.uint8), self.texts[idx]

    def get_augmentations(self, image_width: int, resize_prob: float = 0.75) -> Dict[
        str, transforms.Compose]:
        """
        Initializes augmenting transformations.
        These include a slight rotation, perspective change, random erasing and blurring. Additionally, crops will be
        rescaled randomly to train the model to recognize characters that does not fill the entire image, as well
        as characters, that have been cropped at top or bottom.
        """
        scale = (1 + random.random() * 2)
        resize_to = int(self.image_height // scale), int(image_width // scale)
        pad = self.image_height - resize_to[0]
        return {
            "default": transforms.Compose(
                [
                    transforms.RandomRotation(5),
                    transforms.RandomApply(
                        [
                            transforms.RandomChoice(
                                [
                                    transforms.RandomCrop(
                                        size=resize_to,
                                    ),
                                    transforms.Resize(
                                        (self.image_height, image_width),
                                        antialias=True,
                                    ),
                                    transforms.Compose(
                                        [
                                            transforms.Resize(
                                                resize_to,
                                                antialias=True,
                                            ),
                                            transforms.Pad([0, 0, 0, pad]),
                                        ]
                                    ),
                                ]
                            )
                        ],
                        p=resize_prob,
                    ),
                    transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
                    transforms.RandomErasing(scale=(0.02, 0.1)),
                    transforms.RandomApply(
                        [
                            transforms.Compose(
                                [
                                    transforms.GaussianBlur(5, (0.1, 1.5)),
                                ]
                            ),
                        ],
                        p=0.2,
                    )
                ]
            )
        }
