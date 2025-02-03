"""Dataset class for SSM based OCR training."""

import json
import lzma
import os
import random
import time
from multiprocessing import Queue, Process
from pathlib import Path
from threading import Thread
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
from PIL import Image
from bs4 import BeautifulSoup
from skimage import draw
from torchvision import transforms
from tqdm import tqdm

from src.cgprocess.OCR.shared.tokenizer import OCRTokenizer
from src.cgprocess.OCR.shared.utils import line_has_text, init_tokenizer
from src.cgprocess.shared.utils import xml_polygon_to_polygon_list, get_bbox, enforce_image_limits
from src.cgprocess.shared.datasets import TrainDataset
from src.cgprocess.shared.multiprocessing_handler import run_processes, get_cpu_count


def preprocess_data(image: torch.Tensor, text_lines: List[BeautifulSoup], image_height: int) -> Tuple[
    List[list], List[str]]:
    texts = []
    crops: List[list] = []

    threads = []
    for line in text_lines:
        if line_has_text(line):
            thread = Thread(target=extract_crop,
                            args=(crops, image, line, image_height))
            thread.start()
            threads.append(thread)
            if len(threads) >= 8:
                for thread in threads:
                    thread.join()
                threads = []

            texts.append(line.find('Unicode').text)
    for thread in threads:
        thread.join()
    return crops, texts

# todo: make this stuff in its own class
def extract_crop(crops: List[torch.Tensor], image: torch.Tensor, line: BeautifulSoup, image_height: int) -> None:
    """Crops the image according to bbox information and masks all pixels outside the text line polygon.
    Resizes the crop, so that all crops have the same height.
    Args:
        crops:  list to insert all crops
        image: full input image
        line: current xml object with polygon data
        image_height: fixed height for all crops"""
    region_polygon = enforce_image_limits(torch.tensor(xml_polygon_to_polygon_list(line.Coords["points"])), (image.shape[2], image.shape[1]))

    bbox = get_bbox(region_polygon)
    crop = image.squeeze()[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
    local_polygon = region_polygon.numpy() - np.array([bbox[0], bbox[1]])

    # todo: remove numpy if not necessary

    mask = np.zeros((crop.shape[0] + 1, crop.shape[1] + 1), dtype=np.uint8)
    x_coords, y_coords = draw.polygon(local_polygon[:, 1], local_polygon[:, 0])
    mask[x_coords, y_coords] = 1
    crop *= torch.tensor(mask[:-1, :-1])

    scale = image_height / crop.shape[-2]
    rescale = transforms.Resize((image_height, int(crop.shape[-1] * scale)))
    crops.append(rescale(torch.unsqueeze(crop, 0)).type(torch.uint8).numpy())


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


def extract_page(queue: Queue, paths: Tuple[Path, Path, Path], image_extension: str,
                 cfg: dict) -> None:
    """Extract target and input data for OCR SSM training"""
    tokenizer = init_tokenizer(cfg)
    image_path, annotations_path, target_path = paths
    while True:
        arguments = queue.get()
        if arguments[-1]:
            break
        file_stem, _ = arguments
        start = time.time()
        image, text_lines = load_data(image_path / f"{file_stem}{image_extension}",
                                      annotations_path / f"{file_stem}.xml")
        loaded = time.time()
        print(f"loaded: {loaded - start}")
        crops, texts = preprocess_data(image, text_lines, cfg["image_height"])
        processed = time.time()
        print(f"processed: {processed - loaded}")
        targets = [tokenizer(line).type(torch.uint8).tolist() for line in texts]
        tokenized = time.time()
        print(f"tokenized: {tokenized - processed}")

        json_str = json.dumps({"texts": texts, "targets": targets})  # 2. string (i.e. JSON)
        dumped = time.time()
        print(f"dumped: {dumped - tokenized}")

        json_bytes = json_str.encode('utf-8')  # 3. bytes (i.e. UTF-8)
        bytes = time.time()
        print(f"bytes: {bytes - dumped}")

        with lzma.open(target_path / f"{file_stem}.json", 'wb') as file:  # 4. fewer bytes (i.e. gzip)
            file.write(json_bytes)

        crop_dict = {str(i): crops for i, crops in enumerate(crops)}

        np.savez_compressed(target_path / f"{file_stem}", **crop_dict)
        saved = time.time()
        print(f"saved: {saved - bytes}")




def get_progress(output_path) -> int:
    return len([f for f in os.listdir(output_path) if f.endswith(".json")])


class SSMDataset(TrainDataset):
    """
    Dataset class for SSM based OCR training.
    """

    def __init__(self,
                 kwargs: dict,
                 image_height: int,
                 cfg: dict,
                 num_processes: Optional[int] = None):
        """
        Args:
            image_path: path to folder with images
            target_path: path to folder with xml files
            cfg: configuration file tied to the model
        """
        super().__init__(**kwargs)
        self.image_height = image_height
        self.num_processes = num_processes if num_processes else get_cpu_count() // 8

        self.cfg = cfg

        self.crops: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.texts: List[str] = []

        self.prepare_data()

    def get_data(self):
        for file_stem in tqdm(self.file_stems, desc="Loading data", unit="Pages"):
            with lzma.open(self.target_path / f"{file_stem}.json", 'r') as file:  # 4. gzip
                json_bytes = file.read()  # 3. bytes (i.e. UTF-8)

            json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
            data = json.loads(json_str)

            crops_dict = np.load(self.target_path / f"{file_stem}.npz")
            for i in range(len(crops_dict)):
                self.crops.append(crops_dict[str(i)])


            self.targets.extend(data["targets"])
            self.texts.extend(data["texts"])

    def extract_data(self) -> None:
        """Load ALL xml files and save result image.
        Calls read and draw functions"""

        file_stems = [
            f[:-4] for f in os.listdir(self.annotations_path) if f.endswith(".xml")
        ]

        target_stems = [
            f[:-4] for f in os.listdir(self.target_path) if f.endswith(".npz")
        ]
        path_queue: Queue = Queue()
        total = len(file_stems)

        print(f"num processes: {self.num_processes}")
        processes = [Process(target=extract_page,
                             args=(path_queue, (self.image_path, self.annotations_path, self.target_path), self.image_extension,
                                   self.cfg)) for _ in range(self.num_processes)]

        for file_stem in tqdm(file_stems, desc="Put paths in queue"):
            if file_stem in target_stems:
                continue
            path_queue.put((file_stem, False))

        run_processes({"method": get_progress, "args": self.target_path}, processes, path_queue, total, "Page converting")

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return torch.tensor(self.crops[idx]).float() /255, torch.tensor(self.targets[idx]).long(), self.texts[idx]

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
