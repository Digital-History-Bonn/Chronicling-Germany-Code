"""
module for Dataset class
"""

from __future__ import annotations

import random
from pathlib import Path
from threading import Thread
from typing import Callable, Dict, List, Tuple, Union

import torch

# pylint thinks torch has no name randperm this is wrong
# pylint: disable-next=no-name-in-module
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from cgprocess.layout_segmentation.class_config import PADDING_LABEL
from cgprocess.layout_segmentation.processing.preprocessing import Preprocessing
from cgprocess.layout_segmentation.utils import remove_scaling_errors
from cgprocess.shared.utils import (
    get_file_stems,
    initialize_random_split,
    prepare_file_loading,
)

IMAGE_PATH = "data/images"
TARGET_PATH = "data/targets/"


# todo: implement shared TrainDataset
class CropDataset(Dataset):
    """
    A dataset class for the newspaper page crops used for training, validation and testing.
    """

    def __init__(
        self,
        preprocessing: Preprocessing,
        image_path: str = IMAGE_PATH,
        target_path: str = TARGET_PATH,
        data: Union[List[torch.Tensor], None] = None,
        limit: Union[int, None] = None,
        dataset: str = "transkribus",
        sort: bool = False,
        full_image: bool = False,
        scale_aug: bool = True,
        file_stems: Union[List[str], None] = None,
        name: str = "default",
    ) -> None:
        """
        load images and targets from folder
        :param preprocessing:
        :param data: uses this list instead of loading data from disc
        :param sort: sort file_names for testing purposes
        :param image_path: image path
        :param target_path: target path
        :param limit: limits the quantity of loaded images
        :param dataset: which dataset to expect. Options are 'transkribus' and 'HLNA2013' (europeaner newspaper project)
        :param name: name of dataset type. Eg. train, val and test dataset.
        """

        self.preprocessing = preprocessing
        if full_image:
            preprocessing.crop = False
        self.dataset = dataset
        self.image_path = image_path
        self.target_path = target_path
        self.scale_aug = scale_aug

        self.data: List[torch.Tensor] = []
        if data:
            self.data = data
        else:
            extension, get_file_name = prepare_file_loading(self.dataset)

            if file_stems:
                self.file_stems = file_stems
            else:
                self.file_stems = get_file_stems(extension, Path(image_path))
            if sort:
                self.file_stems.sort()

            if limit is not None:
                assert limit <= len(self.file_stems), (
                    f"Provided limit with size {limit} is greater than the train dataset "
                    f"with size {len(self.file_stems)}."
                )
                self.file_stems = self.file_stems[:limit]

            # iterate over files
            threads = []
            for i, file in enumerate(
                tqdm(self.file_stems, desc=f"preprocessing {name} images", unit="image")
            ):
                thread = Thread(
                    target=self.process_image,
                    args=(
                        dataset,
                        extension,
                        file,
                        get_file_name,
                        image_path,
                        target_path,
                    ),
                )
                thread.start()
                threads.append(thread)
                if i != 0 and i % 1 == 0:
                    for thread in threads:
                        thread.join()
                    threads = []
            for thread in threads:
                thread.join()

        self.augmentations = True

    def process_image(
        self,
        dataset: str,
        extension: str,
        file: str,
        get_file_name: Callable,
        image_path: str,
        target_path: str,
    ) -> None:
        """
        Loads and processes images.
        """
        image, target = self.preprocessing.load(
            f"{image_path}{file}{extension}",
            f"{target_path}{get_file_name(file)}",
            file,
            dataset,
        )
        crops = list(torch.tensor(self.preprocessing(image, target)))
        self.data += crops

    def __len__(self) -> int:
        """
        standard len function
        :return: number of items in dateset
        """
        return len(self.data)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns one datapoint
        :param item: number of the datapoint
        :return (tuple): torch tensor of image, torch tensor of annotation, tuple of mask
        """

        data: torch.Tensor = self.data[item]

        # do augmentations
        if self.augmentations:
            prob = 0.75 if self.scale_aug else 0
            augmentations = self.get_augmentations(prob)
            data = augmentations["default"](data)
            img = augmentations["images"](data[:-1]).float() / 255
            remove_scaling_errors(data[-1])
            # img = (img + (torch.randn(img.shape) * 0.05)).clip(0, 1)     # originally 0.1

            # if random.random() < 0.1:
            #     invert = transforms.RandomInvert(p=1)
            #     img = invert(img)
            #     data[-1][torch.isin(data[-1], torch.tensor([1, 2, 3, 4, 5]))] = 9
            #     data[-1][data[-1] == 9] = 4

        else:
            gray_transform = transforms.Grayscale(num_output_channels=3)
            img = gray_transform(data[:-1]).float() / 255

        return img, data[-1].long()

    def random_split(
        self, ratio: Tuple[float, float, float]
    ) -> Tuple[CropDataset, CropDataset, CropDataset]:
        """
        splits the dataset in parts of size given in ratio
        :param ratio: list[float]:
        :return (tuple): tuple of NewsDatasets
        """
        indices, splits = initialize_random_split(len(self), ratio)
        torch_data = torch.stack(self.data)

        train_dataset = CropDataset(
            self.preprocessing,
            image_path=self.image_path,
            target_path=self.target_path,
            data=list(torch_data[indices[: splits[0]]]),
            dataset=self.dataset,
            scale_aug=self.scale_aug,
        )
        valid_dataset = CropDataset(
            self.preprocessing,
            image_path=self.image_path,
            target_path=self.target_path,
            data=list(torch_data[indices[splits[0] : splits[1]]]),
            dataset=self.dataset,
            scale_aug=self.scale_aug,
        )
        test_dataset = CropDataset(
            self.preprocessing,
            image_path=self.image_path,
            target_path=self.target_path,
            data=list(torch_data[indices[splits[1] :]]),
            dataset=self.dataset,
            scale_aug=self.scale_aug,
        )

        return train_dataset, valid_dataset, test_dataset

    def get_augmentations(
        self, resize_prob: float = 0.75
    ) -> Dict[str, transforms.Compose]:
        """Defines transformations
        :param resize_prob: Probability of resize. This allows to deactivate the scaling augmentation.
        """
        resize_to = int(self.preprocessing.crop_size // (1 + random.random() * 2))
        pad = self.preprocessing.crop_size - resize_to
        return {
            "default": transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(20),
                    transforms.RandomApply(
                        [
                            transforms.RandomChoice(
                                [
                                    transforms.RandomResizedCrop(
                                        size=self.preprocessing.crop_size,
                                        scale=(0.33, 1.0),
                                    ),
                                    transforms.Compose(
                                        [
                                            transforms.Resize(
                                                resize_to,
                                                antialias=True,
                                            ),
                                            transforms.Pad([0, 0, pad, pad], fill=PADDING_LABEL),
                                        ]
                                    ),
                                ]
                            )
                        ],
                        p=resize_prob,
                    ),
                    transforms.RandomPerspective(p=0.2),
                ]
            ),
            "images": transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.RandomErasing(value=PADDING_LABEL),
                    transforms.RandomApply(
                        [
                            transforms.Compose(
                                [
                                    transforms.GaussianBlur(9, (0.1, 1.5)),
                                ]
                            ),
                        ],
                        p=0.2,
                    ),
                ]
            ),
        }  # originally 0.8
