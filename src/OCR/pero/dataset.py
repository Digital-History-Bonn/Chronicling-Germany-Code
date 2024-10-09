"""Dataset class for Pero Transformer based OCR."""

import glob
import os
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.OCR.pero.config import PAD_HEIGHT, PAD_WIDTH, MAX_SEQUENCE_LENGTH, ALPHABET
from src.OCR.pero.tokenizer import Tokenizer
from src.OCR.utils import get_bbox, load_image


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for Transformer based OCR.
    """
    def __init__(self, image_path: str,
                 target_path: str,
                 pad_seq: bool = False,
                 cache_images: bool = False):
        """
        
        Args:
            image_path: path to folder with images
            target_path: path to folder with xml files
            pad_seq: boolean to activate padding for sequences
            cache_images: load all images into cache (needs a lot of RAM!) 
        """
        self.image_path = image_path
        self.target_path = target_path
        self.cache_images = cache_images

        self.tokenizer = Tokenizer(ALPHABET, pad=pad_seq, max_length=MAX_SEQUENCE_LENGTH)

        self.images: List[str] = []
        self.bboxes: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.texts: List[str] = []
        self.image_dict: Dict[str, torch.Tensor] = {}

        self.get_data()

    def get_data(self) -> None:
        """Loads the data by reading the annotation."""
        image_paths = list(glob.glob(os.path.join(self.image_path, '*.jpg')))
        xml_paths = [f"{x[:-4]}.xml" for x in image_paths]

        for image_path, xml_path in tqdm(zip(image_paths, xml_paths),
                                         total=len(image_paths),
                                         desc='loading dataset'):

            bboxes, texts = read_xml(xml_path)
            self.bboxes.extend(bboxes)
            self.texts.extend(texts)
            self.targets.extend([self.tokenizer(line) for line in texts])
            self.images.extend([image_path] * len(bboxes))

            if image_path not in self.image_dict.keys() and self.cache_images:
                self.image_dict[image_path] = load_image(image_path)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.cache_images:
            image = self.image_dict[self.images[idx]]
        else:
            image = load_image(self.images[idx])

        bbox = self.bboxes[idx]
        target = self.targets[idx]
        text = self.texts[idx]

        # pylint: disable=duplicate-code
        crop = image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

        pad_height = max(0, PAD_HEIGHT - crop.shape[1])
        pad_width = max(0, PAD_WIDTH - crop.shape[2])
        crop = F.pad(crop, (pad_width, 0, pad_height, 0), "constant", 0)
        crop = crop[:, :PAD_HEIGHT]

        return crop.float() / 255, target, text


def read_xml(xml_path: str) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Reads the xml files.
    Args:
        xml_path: path to xml file with annotations.

    Returns:
        bboxes: bounding boxes text lines.
        texts: text of text lines.
    """
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    bboxes = []
    texts = []

    text_lines = page.find_all('TextLine')
    for line in text_lines:
        coords = line.find('Coords')
        region_polygon = torch.tensor([tuple(map(int, point.split(','))) for
                                       point in coords['points'].split()])
        bboxes.append(torch.tensor(get_bbox(region_polygon)))
        texts.append(line.find('Unicode').text)

    return bboxes, texts


if __name__ == '__main__':
    dataset = Dataset(image_path='data/preprocessedOCR/train',
                      target_path='data/preprocessedOCR/train',
                      cache_images=True)
