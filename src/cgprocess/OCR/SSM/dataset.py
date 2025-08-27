"""Dataset class for SSM based OCR training."""

import json
import lzma
import os
from multiprocessing import Process, Queue
from pathlib import Path
from threading import Thread
from typing import List, Optional, Tuple

import numpy as np
import torch
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import Resize
from tqdm import tqdm
from skimage.draw import line as draw_line

from cgprocess.OCR.shared.utils import init_tokenizer, line_has_text
from cgprocess.shared.datasets import TrainDataset
from cgprocess.shared.multiprocessing_handler import get_cpu_count, run_processes
from cgprocess.shared.utils import (
    enforce_image_limits,
    get_bbox,
    xml_polygon_to_polygon_list,
)


def preprocess_data(
    image: torch.Tensor,
    text_lines: List[BeautifulSoup],
    image_height: int,
    predict: bool = False,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Extract crop for each text-line and append ids or textual data if needed."""
    texts = []
    crops: List[np.ndarray] = []
    ids = []

    for line in text_lines:
        if line_has_text(line) or predict:
            valid = extract_crop(crops, image, line, image_height)
            if not valid:
                continue
            if predict:
                ids.append(line["id"])
                if line_has_text(line):
                    texts.append(line.find("Unicode").text)  # type: ignore
            else:
                texts.append(line.find("Unicode").text)  # type: ignore

    return crops, texts, ids  # type: ignore


# def descew():


def extract_crop(
    crops: List[np.ndarray], image: torch.Tensor, line: BeautifulSoup, crop_height: int
) -> bool:
    """Crops the image according to bbox information and masks all pixels outside the text line polygon.
    Resizes the crop, so that all crops have the same height.
    Args:
        crops:  list to insert all crops
        image: full input image
        line: current xml object with polygon data
        crop_height: fixed height for all crops"""
    assert image.dtype == torch.uint8

    region_polygon = enforce_image_limits(  # type: ignore
        torch.tensor(xml_polygon_to_polygon_list(line.Coords["points"])),  # type: ignore
        (image.shape[2], image.shape[1]),
    )  # type: ignore

    # initialize
    bbox = get_bbox(region_polygon)
    crop = image.squeeze()[bbox[1] : bbox[3] + 1, bbox[0] : bbox[2] + 1].clone()  # type: ignore

    local_polygon = region_polygon.numpy() - np.array([bbox[0], bbox[1]])

    img = Image.new("1", (crop.shape[0] + 1, crop.shape[1] + 1), 0)
    draw_polygon = ImageDraw.Draw(img)
    draw_polygon.polygon(list(map(tuple, np.roll(local_polygon, 1, axis=1))), fill=1)

    transform = transforms.PILToTensor()
    mask = torch.permute(transform(img), (0, 2, 1)).type(torch.uint8)

    if crop.shape[-1] <= 0 or crop.shape[-2] <= 0:
        return False

    crop, rescale, scale = scale_crop(crop, crop_height)
    mask = rescale(mask[:, :-1, :-1])

    if crop.shape[-1] < crop_height:
        return False

    baseline = None
    if line.find("Baseline"):
        raw_baseline = enforce_image_limits(  # type: ignore
            torch.tensor(xml_polygon_to_polygon_list(line.Baseline["points"])),  # type: ignore
            (image.shape[2], image.shape[1]),
        )  # type: ignore
        local_baseline = raw_baseline.numpy() - np.array([bbox[0], bbox[1]])
        baseline = (local_baseline * scale).astype(int)
        if baseline[0][0] != 0:
            baseline = np.insert(baseline, 0, np.array([[0, baseline[0][1]]]), axis=0)
        if baseline[-1][0] != crop.shape[-1]-1:
            baseline = np.append(baseline, np.array([[crop.shape[-1]-1, baseline[-1][1]]]), axis=0)

    crop *= mask

    # todo: do this before scaling, so that we do not have to rescale
    raw_shifts = []
    for i in range(baseline.shape[0] - 1): # todo: remove double shift entries properly
        current = baseline[i]
        next = baseline[i+1]
        rr, cc = draw_line(current[0], current[1], next[0], next[1])
        if i < baseline.shape[0] - 2:
            raw_shifts.append(cc[:-1])
        else:
            raw_shifts.append(cc)

    crop = crop.squeeze()
    shifts = np.concat(raw_shifts)
    shifts = shifts - shifts.min()

    max_shift = shifts.max()
    if max_shift > 30:
        pass

    rows = np.arange(crop.shape[0])[:, None]

    shifted_rows = (rows+shifts[None, :]) % crop.shape[0]
    if shifted_rows.shape[1] != crop.shape[1]: # todo: remove double shift entries properly
        shifted_rows = shifted_rows[:, :crop.shape[1]]
    crop = crop[shifted_rows, np.arange(crop.shape[1])[None, :]]

    if max_shift > 0:
        crop = crop[:-max_shift, :]

    crop, _, _ = scale_crop(crop, crop_height)
    #
    # transform = transforms.ToPILImage()
    # img = transform(crop)
    #
    # img.save("test_baseline_deskew_scale.png")

    crops.append(crop.numpy())
    return True


def scale_crop(crop: Tensor, crop_height: int) -> Tuple[Tensor, Resize, float]:
    """
    Scale crop, sush that it has the desired height and preserves aspect ratio.
    Args:
        crop: image data
        crop_height: desired height

    Returns: scaled crop and Resize transform as well as scale value
    """
    scale = crop_height / crop.shape[-2]
    rescale = transforms.Resize((crop_height, int(crop.shape[-1] * scale)))
    crop = rescale(torch.unsqueeze(crop, 0))
    return crop, rescale, scale


def load_data(
    image_path: Path, xml_path: Path
) -> Tuple[torch.Tensor, List[BeautifulSoup]]:
    """Load image and xml data, transform the PIL image to a torch tensor and extract all xml text line objects."""
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()
    pil_image = Image.open(image_path).convert("L")
    transform = transforms.PILToTensor()
    image = transform(pil_image)
    # Parse the XML data
    soup = BeautifulSoup(data, "xml")
    page = soup.find("Page")
    text_lines = page.find_all("TextLine")  # type: ignore
    return image, text_lines


def extract_page(
    queue: Queue,
    paths: Tuple[Path, Path, Path],
    image_extension: str,
    cfg: dict,
    result_Queue: Optional[Queue] = None,
    predict: bool = False,
) -> None:
    """Extract target and input data for OCR SSM training"""
    tokenizer = init_tokenizer(cfg)
    image_path, annotations_path, target_path = paths
    while True:
        arguments = queue.get()
        if arguments[-1]:
            break
        file_stem, _ = arguments
        image, text_lines = load_data(
            image_path / f"{file_stem}{image_extension}",
            annotations_path / f"{file_stem}.xml",
        )
        crops, texts, ids = preprocess_data(
            image, text_lines, cfg["preprocessing"]["image_height"], predict
        )
        targets = [tokenizer(line).type(torch.uint8).tolist() for line in texts]

        json_str = json.dumps({"texts": texts, "targets": targets, "ids": ids})

        json_bytes = json_str.encode("utf-8")

        with lzma.open(target_path / f"{file_stem}.json", "wb") as file:
            file.write(json_bytes)

        crop_dict = {str(i): crops for i, crops in enumerate(crops)}

        np.savez_compressed(target_path / f"{file_stem}", **crop_dict)

        if result_Queue:
            result_Queue.put((file_stem, annotations_path, target_path, False))


def get_progress(output_path: Path) -> int:
    """Get progress by counting number of json files in output path."""
    return len([f for f in os.listdir(output_path) if f.endswith(".json")])


class SSMDataset(TrainDataset): #type: ignore
    """
    Dataset class for SSM based OCR training.
    """

    def __init__(
        self,
        kwargs: dict,
        crop_height: int,
        cfg: dict,
        num_processes: Optional[int] = None,
        augmentation: bool = False,
    ):
        """
        Args:
            kwargs: arguments for super class
            crop_height: Fixed height. All crops will be scaled to this height
            cfg: configuration file tied to the model
        """
        super().__init__(**kwargs)
        self.augmentation = augmentation
        self.image_height = crop_height
        # self.num_processes = num_processes if num_processes else get_cpu_count() // 8
        self.num_processes = 1

        self.cfg = cfg

        self.data: List[Tuple[np.ndarray, torch.Tensor, str]] = []

        self.prepare_data()

    def get_data(self) -> None:
        """Start threads for loading preprocessed data."""
        threads = []
        for file_stem in tqdm(self.file_stems, desc="Loading data", unit="Pages"):

            thread = Thread(target=self.load_preprocessed_data, args=[file_stem])
            thread.start()
            threads.append(thread)
            if len(threads) >= 16:
                for thread in threads:
                    thread.join()
                threads = []
        for thread in threads:
            thread.join()

    def load_preprocessed_data(self, file_stem: str) -> None:
        """Load crops, targets and texts and append them as tuple to the self.data list."""
        with lzma.open(self.target_path / f"{file_stem}.json", "r") as file:  # 4. gzip
            json_bytes = file.read()  # 3. bytes (i.e. UTF-8)
        json_str = json_bytes.decode("utf-8")  # 2. string (i.e. JSON)
        data = json.loads(json_str)
        crops_dict = np.load(self.target_path / f"{file_stem}.npz")
        for i in range(len(crops_dict)):
            self.data.append((crops_dict[str(i)], data["targets"][i], data["texts"][i]))

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

        processes = [
            Process(
                target=extract_page,
                args=(
                    path_queue,
                    (self.image_path, self.annotations_path, self.target_path),
                    self.image_extension,
                    self.cfg,
                ),
            )
            for _ in range(self.num_processes)
        ]

        for file_stem in tqdm(file_stems, desc="Put paths in queue"):
            if file_stem in target_stems:
                continue
            path_queue.put((file_stem, False))

        run_processes(
            {"method": get_progress, "args": [self.target_path]},
            processes,
            path_queue,
            total,
            "Page converting",
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        crop, target, text = self.data[idx]
        data = torch.tensor(crop).float()
        if self.augmentation:
            # augment = self.get_augmentations(data.shape[-1])
            augment = self.get_augmentations()
            data = augment(data)
        return data / 255, torch.tensor(target).long(), text

    def get_augmentations(self) -> transforms.Compose:
        # def get_augmentations(self, image_width: int, resize_prob: float = 0.2,
        #                       kernel_size: int = 5) -> transforms.Compose:
        """
        Initializes augmenting transformations.
        These include a slight rotation, perspective change, random erasing and blurring. Additionally, crops will be
        rescaled randomly to train the model to recognize characters that does not fill the entire image, as well
        as characters, that have been cropped at top or bottom.
        Args:
            kernel_size: kernel_size for gaussian blurring
        """
        # pad_kernel = kernel_size if image_width <= kernel_size else 0
        # scale = (1 + random.random() * 1)
        # resize_to = int(self.image_height // scale), int(image_width // scale) + 1
        # crop_size = resize_to[0], image_width
        # pad_y = self.image_height - resize_to[0]
        # pad_x = kernel_size - resize_to[1] if resize_to[1] < kernel_size else 0
        # pad_x = kernel_size - image_width if image_width < kernel_size else pad_x
        return transforms.Compose(
            [
                # transforms.Pad((pad_kernel, 0, 0, 0)),
                transforms.RandomApply(
                    [transforms.GaussianBlur(5, (0.1, 1.5))],
                    p=0.2,
                ),
                transforms.RandomErasing(scale=(0.02, 0.1), p=0.2),
                # transforms.RandomApply(
                #     [
                #         transforms.RandomChoice(
                #             [
                #                 transforms.Compose([
                #                     transforms.RandomCrop(
                #                         size=crop_size,
                #                     ),
                #                     transforms.Resize(
                #                         (self.image_height, int(image_width * scale)),
                #                         antialias=True,
                #                     ), ]),
                #                 transforms.Compose(
                #                     [
                #                         transforms.Resize(
                #                             resize_to,
                #                             antialias=True,
                #                         ),
                #                         transforms.RandomChoice(
                #                             [
                #                                 transforms.Pad((pad_x, pad_y, 0, 0)),
                #                                 transforms.Pad((0, 0, pad_x, pad_y))
                #                             ]),
                #                     ]
                #                 ),
                #             ]
                #         )
                #     ],
                #     p=resize_prob,
                # ),
        # self._transforms = Compose([
        #     ToFloat(),
        #     PixelDropout(p=0.2),
        #     OneOf([
        #         MotionBlur(p=0.2),
        #         MedianBlur(blur_limit=3, p=0.1),
        #         Blur(blur_limit=3, p=0.1),
        #     ], p=0.2),
        #     OneOf([
        #         OpticalDistortion(p=0.3),
        #         ElasticTransform(alpha=7, sigma=25, p=0.1),
        #         SafeRotate(limit=(-3, 3), border_mode=cv2.BORDER_CONSTANT, p=0.2)
        #     ], p=0.2),
        # ], p=0.5)
            ]
        )
