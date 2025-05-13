"""Dataset class for Pero Transformer based OCR."""

import glob
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from albumentations import Blur, Compose, ElasticTransform, MedianBlur, MotionBlur, OneOf, \
    SafeRotate, OpticalDistortion, PixelDropout, ToFloat

from src.cgprocess.OCR.shared.tokenizer import OCRTokenizer
from src.cgprocess.OCR.shared.utils import load_image, read_xml
from src.cgprocess.OCR.Transformer.config import (
    ALPHABET,
    MAX_SEQUENCE_LENGTH,
    PAD_HEIGHT,
    PAD_WIDTH,
)


class DefaultAugmenter:
    """Default Augmenter form kraken.

    Source: https://github.com/mittagessen/kraken/blob/main/kraken/lib/dataset/recognition.py

    """

    def __init__(self):
        cv2.setNumThreads(0)

        self._transforms = Compose([
            ToFloat(),
            PixelDropout(p=0.2),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            OneOf([
                OpticalDistortion(p=0.3),
                ElasticTransform(alpha=7, sigma=25, p=0.1),
                SafeRotate(limit=(-3, 3), border_mode=cv2.BORDER_CONSTANT, p=0.2)
            ], p=0.2),
        ], p=0.5)

    def __call__(self, image):
        im = image.permute((1, 2, 0)).numpy()
        o = self._transforms(image=im)

        return torch.tensor(o['image'].transpose(2, 0, 1))


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for Transformer based OCR.
    """

    def __init__(self,
                 image_path: str,
                 target_path: str,
                 augmentations: bool = True,
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
        self.augmentations = DefaultAugmenter() if augmentations else None
        self.cache_images = cache_images

        self.tokenizer = OCRTokenizer(
            ALPHABET, pad=pad_seq, max_length=MAX_SEQUENCE_LENGTH
        )

        self.images: List[str] = []
        self.bboxes: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.texts: List[str] = []
        self.image_dict: Dict[str, torch.Tensor] = {}

        self.get_data()

    def get_data(self) -> None:
        """Loads the data by reading the annotation."""
        image_paths = list(glob.glob(os.path.join(self.image_path, "*.jpg")))
        xml_paths = [f"{x[:-4]}.xml" for x in image_paths]

        for image_path, xml_path in tqdm(zip(image_paths, xml_paths),
                                         total=len(image_paths),
                                         desc='loading dataset'):

            bboxes, texts, _ = read_xml(xml_path)
            self.bboxes.extend(bboxes)
            self.texts.extend(texts)
            self.targets.extend([self.tokenizer(line) for line in texts])
            self.images.extend([image_path] * len(bboxes))

            if (
                image_path not in self.image_dict.keys() and self.cache_images
            ):  # pylint: disable=consider-iterating-dictionary
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
        crop = image[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]

        pad_height = max(0, PAD_HEIGHT - crop.shape[1])
        pad_width = max(0, PAD_WIDTH - crop.shape[2])
        crop = F.pad(crop, (pad_width, 0, pad_height, 0), "constant", 0)
        crop = crop[:, :PAD_HEIGHT]

        if self.augmentations:
            crop = self.augmentations(crop)

        return crop.float(), target, text


if __name__ == "__main__":
    dataset = Dataset(
        image_path="data/preprocessedOCR/train",
        target_path="data/preprocessedOCR/train",
        cache_images=True,
    )
