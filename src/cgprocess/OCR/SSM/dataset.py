"""Dataset class for SSM based OCR training."""

import json
import lzma
import os
import random
from multiprocessing import Queue, Process, set_start_method
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


def preprocess_data(image: torch.Tensor, text_lines: List[BeautifulSoup], image_height: int, predict: bool = False) -> Tuple[
    List[list], List[str], List[str]]:
    texts = []
    crops: List[list] = []
    ids = []

    threads = []
    for line in text_lines:
        if line_has_text(line) or predict:
            thread = Thread(target=extract_crop,
                            args=(crops, image, line, image_height))
            thread.start()
            threads.append(thread)
            if len(threads) >= 8:
                for thread in threads:
                    thread.join()
                threads = []

            if predict:
                ids.append(line["id"])
            else:
                texts.append(line.find('Unicode').text)

    for thread in threads:
        thread.join()
    return crops, texts, ids


# todo: make this stuff in its own class
def extract_crop(crops: List[torch.Tensor], image: torch.Tensor, line: BeautifulSoup, crop_height: int) -> None:
    """Crops the image according to bbox information and masks all pixels outside the text line polygon.
    Resizes the crop, so that all crops have the same height.
    Args:
        crops:  list to insert all crops
        image: full input image
        line: current xml object with polygon data
        crop_height: fixed height for all crops"""
    assert image.dtype == torch.uint8

    region_polygon = enforce_image_limits(torch.tensor(xml_polygon_to_polygon_list(line.Coords["points"])),
                                          (image.shape[2], image.shape[1]))

    # initialize
    bbox = get_bbox(region_polygon)
    crop = image.squeeze()[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]

    local_polygon = region_polygon.numpy() - np.array([bbox[0], bbox[1]])

    # create mask
    mask = np.zeros((crop.shape[0] + 1, crop.shape[1] + 1), dtype=np.uint8)
    x_coords, y_coords = draw.polygon(local_polygon[:, 1], local_polygon[:, 0])
    mask[x_coords, y_coords] = 1

    # scale to crop_height
    scale = crop_height / crop.shape[-2]
    rescale = transforms.Resize((crop_height, int(crop.shape[-1] * scale)))
    crop = rescale(torch.unsqueeze(crop, 0))
    mask = rescale(torch.unsqueeze(torch.tensor(mask[:-1, :-1]), 0))

    if crop.shape[-1] < crop_height:
        return

    # apply mask
    crop *= mask
    crops.append(crop.numpy())


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
                 cfg: dict, result_Queue: Optional[Queue] = None, predict: bool = False) -> None:
    """Extract target and input data for OCR SSM training"""
    tokenizer = init_tokenizer(cfg)
    image_path, annotations_path, target_path = paths
    while True:
        arguments = queue.get()
        if arguments[-1]:
            break
        file_stem, _ = arguments
        image, text_lines = load_data(image_path / f"{file_stem}{image_extension}",
                                      annotations_path / f"{file_stem}.xml")
        crops, texts, ids = preprocess_data(image, text_lines, cfg["image_height"], predict)
        targets = [tokenizer(line).type(torch.uint8).tolist() for line in texts]

        json_str = json.dumps({"texts": texts, "targets": targets, "ids": ids})

        json_bytes = json_str.encode('utf-8')

        with lzma.open(target_path / f"{file_stem}.json", 'wb') as file:
            file.write(json_bytes)

        crop_dict = {str(i): crops for i, crops in enumerate(crops)}

        np.savez_compressed(target_path / f"{file_stem}", **crop_dict)

        if result_Queue:
            result_Queue.put((file_stem, annotations_path, target_path, False), block=True)


def get_progress(output_path) -> int:
    return len([f for f in os.listdir(output_path) if f.endswith(".json")])


class SSMDataset(TrainDataset):
    """
    Dataset class for SSM based OCR training.
    """

    def __init__(self,
                 kwargs: dict,
                 crop_height: int,
                 cfg: dict,
                 num_processes: Optional[int] = None,
                 augmentation: bool = False):
        """
        Args:
            kwargs: arguments for super class
            crop_height: Fixed height. All crops will be scaled to this height
            cfg: configuration file tied to the model
        """
        super().__init__(**kwargs)
        self.augmentation = augmentation
        self.image_height = crop_height
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

        set_start_method('spawn')
        print(f"num processes: {self.num_processes}")
        processes = [Process(target=extract_page,
                             args=(path_queue, (self.image_path, self.annotations_path, self.target_path),
                                   self.image_extension,
                                   self.cfg)) for _ in range(self.num_processes)]

        for file_stem in tqdm(file_stems, desc="Put paths in queue"):
            if file_stem in target_stems:
                continue
            path_queue.put((file_stem, False))

        run_processes({"method": get_progress, "args": [self.target_path]}, processes, path_queue, total,
                      "Page converting")

        set_start_method('fork')

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        data = torch.tensor(self.crops[idx]).float()
        if self.augmentation:
            augment = self.get_augmentations(data.shape[-1])
            data = augment(data)
        return data, torch.tensor(self.targets[idx]).long(), self.texts[idx]

    def get_augmentations(self, image_width: int, resize_prob: float = 0.3, kernel_size: int = 5) -> transforms.Compose:
        """
        Initializes augmenting transformations.
        These include a slight rotation, perspective change, random erasing and blurring. Additionally, crops will be
        rescaled randomly to train the model to recognize characters that does not fill the entire image, as well
        as characters, that have been cropped at top or bottom.
        Args:
            kernel_size: kernel_size for gaussian blurring
        """
        pad_kernel = kernel_size if image_width <= kernel_size else 0
        scale = (1 + random.random() * 1)
        resize_to = int(self.image_height // scale), int(image_width // scale) + 1
        crop_size = resize_to[0], image_width
        pad_y = self.image_height - resize_to[0]
        pad_x = kernel_size - resize_to[1] if resize_to[1] < kernel_size else 0
        pad_x = kernel_size - image_width if image_width < kernel_size else pad_x
        return transforms.Compose(
            [
                transforms.Pad((pad_kernel, 0, 0, 0)),
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(5, (0.1, 1.5))
                    ],
                    p=0.1,
                ),
                # transforms.RandomErasing(scale=(0.02, 0.1)),
                transforms.RandomApply(
                    [
                        transforms.RandomChoice(
                            [
                                transforms.Compose([
                                    transforms.RandomCrop(
                                        size=crop_size,
                                    ),
                                    transforms.Resize(
                                        (self.image_height, int(image_width * scale)),
                                        antialias=True,
                                    ), ]),
                                transforms.Compose(
                                    [
                                        transforms.Resize(
                                            resize_to,
                                            antialias=True,
                                        ),
                                        transforms.RandomChoice(
                                            [
                                                transforms.Pad((pad_x, pad_y, 0, 0)),
                                                transforms.Pad((0, 0, pad_x, pad_y))
                                            ]),
                                    ]
                                ),
                            ]
                        )
                    ],
                    p=resize_prob,
                ),
            ]
        )
