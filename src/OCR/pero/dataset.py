"""Dataset class for Pero Transformer based OCR."""

import glob
import os
from typing import List

import torch
import torch.nn.functional as F
from skimage import io
from tqdm import tqdm

from src.OCR.pero.utils import Tokenizer, read_xml


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path: str,
                 target_path: str,
                 alphabet: List[str],
                 pad: bool = False,
                 max_seq_len: int = 512,
                 pad_height: int = 64,
                 pad_width: int = 16,
                 cache_images: bool = False):
        self.image_path = image_path
        self.target_path = target_path
        self.pad_height = pad_height
        self.pad_width = pad_width
        self.cache_images = cache_images

        self.tokenizer = Tokenizer(alphabet, pad=pad, max_lenght=max_seq_len)

        self.images = []
        self.bboxes = []
        self.targets = []
        self.texts = []
        self.image_dict = {}

        self.get_data()
        self.len = len(self.images)

    def get_data(self):
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
                self.image_dict[image_path] = torch.tensor(io.imread(image_path)).permute(2, 0, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.cache_images:
            image = self.image_dict[self.images[idx]]
        else:
            image = torch.tensor(io.imread(self.images[idx])).permute(2, 0, 1)

        bbox = self.bboxes[idx]
        target = self.targets[idx]
        text = self.texts[idx]

        crop = image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

        pad_height = max(0, self.pad_height - crop.shape[1])
        pad_width = max(0, self.pad_width - crop.shape[2])
        crop = F.pad(crop, (pad_width, 0, pad_height, 0), "constant", 0)
        crop = crop[:, :self.pad_height]

        return crop.float() / 255, target, text


if __name__ == '__main__':
    from src.OCR.pero.trainer import ALPHABET

    dataset = Dataset(image_path='data/preprocessedOCR/train',
                      target_path='data/preprocessedOCR/train',
                      alphabet=ALPHABET,
                      cache_images=True)
