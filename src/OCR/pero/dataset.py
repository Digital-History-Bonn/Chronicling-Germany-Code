import glob
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from skimage import io
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.OCR.pero.utils import Tokenizer
from src.OCR.utils import get_bbox


def read_xml(xml_path: str) -> Tuple[List[torch.Tensor], List[str]]:
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
        image_paths = list(glob.glob(os.path.join(self.image_path, '*.jpg')))[:200]
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
    import matplotlib.pyplot as plt

    ALPHABET = ['<PAD>', '<START>', '<NAN>', '<END>',
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                'ä', 'ö', 'ü', 'ſ', 'ẞ', 'à',
                '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                ' ', ',', '.', '?', '!', '-', '_', ':', ';', '/', '(', ')',
                '\"', '\'', '&', '+' '~']

    dataset = Dataset(image_path='../data/preprocessedOCR/train',
                      target_path='../data/preprocessedOCR/train',
                      alphabet=ALPHABET,
                      cache_images=True)
    crop, text = dataset[0]

    print(f"{crop.shape=}")
    print(f"{text=}")

    plt.imshow(crop.permute(1, 2, 0).int())
    plt.title(text)
    plt.show()
