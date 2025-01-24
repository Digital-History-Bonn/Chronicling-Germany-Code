"""Dataset class for SSM based OCR training."""

import glob
import os
import random
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
from PIL import Image
from bs4 import BeautifulSoup
from skimage import draw
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from src.cgprocess.OCR.shared.tokenizer import Tokenizer
from src.cgprocess.OCR.shared.utils import get_bbox, line_has_text
from src.cgprocess.layout_segmentation.processing.read_xml import xml_polygon_to_polygon_list


def preprocess_data(image: torch.tensor, text_lines: List[BeautifulSoup], image_height: int):
    texts = []
    crops = []
    for line in text_lines:
        if line_has_text(line):
            extract_crop(crops, image, line, image_height)

            texts.append(line.find('Unicode').text)
    return crops, texts


def extract_crop(crops:List[torch.tensor], image: torch.tensor, line: List[BeautifulSoup], image_height: int):
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
    crops.append(rescale(crop))


def load_data(image_path: Path, xml_path: Path):
    """Load image and xml data, transform the PIL image to a torch tensor and extract all xml text line objects."""
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()
    pil_image = Image.open(image_path).convert('L')
    transform = transforms.PILToTensor()
    image = transform(pil_image).float() / 255
    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    text_lines = page.find_all('TextLine')
    return image, text_lines


class TrainDataset(Dataset):
    """
    Dataset class for SSM based OCR training.
    """

    def __init__(self, image_path: Path,
                 target_path: Path,
                 cfg: dict):
        """
        Args:
            image_path: path to folder with images
            target_path: path to folder with xml files
            cfg: configuration file tied to the model
        """
        self.image_path = image_path
        self.target_path = target_path

        self.tokenizer = Tokenizer(**cfg["tokenizer"])
        self.image_height = cfg["image_height"]

        self.crops: List[torch.Tensor] = []
        self.bboxes: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.texts: List[str] = []

        self.get_data()

    def get_data(self) -> None:
        """Loads image and corresponding text lines with ground truth text and performs cropping and masking for all
        lines."""
        image_paths = list(glob.glob(os.path.join(self.image_path, '*.jpg')))
        xml_paths = [f"{x[:-4]}.xml" for x in image_paths]

        for image_path, xml_path in tqdm(zip(image_paths, xml_paths),
                                         total=len(image_paths),
                                         desc='loading dataset'):

            image, text_lines = load_data(image_path, xml_path)
            crops, texts = preprocess_data(image, text_lines, self.image_height)

            self.texts.extend(texts)
            self.targets.extend([self.tokenizer(line) for line in texts])
            self.crops.extend(crops)

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return self.crops[idx], self.targets[idx], self.texts[idx]

    def get_augmentations(self, image_width: int, image_height: int, resize_prob: float = 0.75) -> Dict[str, transforms.Compose]:
        """
        Initializes augmenting transformations.
        These include a slight rotation, perspective change, random erasing and blurring. Additionally, crops will be
        rescaled randomly to train the model to recognize characters that does not fill the entire image, as well
        as characters, that have been cropped at top or bottom.
        """
        scale = (1 + random.random() * 2)
        resize_to = int(image_height // scale), int(image_width // scale)
        pad = image_height - resize_to[0]
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
                                        (image_height, image_width),
                                        antialias=True,
                                    ),
                                    transforms.Compose(
                                        [
                                            transforms.Resize(
                                                resize_to,
                                                antialias=True,
                                            ),
                                            transforms.Pad([0,0,0,pad]),
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
