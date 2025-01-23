"""Dataset class for SSM based OCR training."""

import glob
import os
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
from PIL import Image
from bs4 import BeautifulSoup
from skimage import draw
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from src.cgprocess.OCR.Transformer import PAD_HEIGHT, PAD_WIDTH, MAX_SEQUENCE_LENGTH, ALPHABET
from src.cgprocess.OCR.shared.tokenizer import Tokenizer
from src.cgprocess.OCR.shared.utils import get_bbox, load_image, read_xml, line_has_text
from src.cgprocess.layout_segmentation.processing.read_xml import xml_polygon_to_polygon_list


def preprocess_data(crops, image, text_lines, texts):
    for line in text_lines:
        if line_has_text(line):
            preprocess_image(crops, image, line)

            texts.append(line.find('Unicode').text)


def preprocess_image(crops, image, line):
    region_polygon = torch.tensor(xml_polygon_to_polygon_list(line.Coords["points"]))
    bbox = torch.tensor(get_bbox(region_polygon))
    crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    mask = torch.zeros_like(crop)
    mask[draw.polygon(region_polygon[:, 1], region_polygon[:, 0])] = 1
    crop *= mask
    crops.append(crop)


def load_data(image_path, xml_path):
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()
    pil_image = Image.open(image_path).convert('L')
    transform = transforms.PILToTensor()
    image = transform(pil_image).float() / 255
    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    texts = []
    crops = []
    text_lines = page.find_all('TextLine')
    return crops, image, text_lines, texts


class TrainDataset(Dataset):
    """
    Dataset class for Transformer based OCR.
    """

    def __init__(self, image_path: str,
                 target_path: str,
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

            crops, image, text_lines, texts = load_data(image_path, xml_path)
            preprocess_data(crops, image, text_lines, texts)

            self.texts.extend(texts)
            self.targets.extend([self.tokenizer(line) for line in texts])
            self.crops.extend(crops)

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return self.crops[idx], self.targets[idx], self.texts[idx]
