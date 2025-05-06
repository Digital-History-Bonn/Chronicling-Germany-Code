"""Dataset class for Transformer baseline detection model training."""

import glob
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):  # type: ignore
    """Newspaper Class for training."""

    def __init__(
        self,
        data_path: str,
        augmentations: Optional[Module] = None,
        cropping: bool = True,
    ) -> None:
        """
        Newspaper Class for training.

        Args:
            data_path: path to folder with images
            target_path: path to folder with targets
            augmentations: torchvision transforms for on-the-fly augmentations
            cropping: whether to crop images or not
        """
        super().__init__()
        self.data_path = data_path
        self.data = [x.split(os.sep)[-1][:-4] for x in glob.glob(f"{data_path}/*.jpg")]
        self.cropping = cropping
        self.crop_size = (256, 256)
        self.augmentations = augmentations

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns image and target (boxes, labels, img_number) from dataset.

        Args:
            index: index of datapoint

        Returns:
            image, target
        """
        image = torch.tensor(io.imread(f"{self.data_path}/{self.data[index]}.jpg"))
        image = image.permute(2, 0, 1) / 256
        target = torch.tensor(
            np.load(f"{self.data_path}/{self.data[index]}.npz")["array"]
        )
        target = target.permute(2, 0, 1).float()

        # mask image
        image = image * target[None, 5, :, :]
        target = target[:4, :, :]

        _, width, height = image.shape
        resize = transforms.Resize((width // 2, height // 2))
        image = resize(image)
        target = F.max_pool2d(target, 2)

        if self.augmentations and random.random() < 0.5:
            resize = transforms.Resize((width // 4, height // 4))
            image = resize(image)
            target = F.max_pool2d(target, 2)

        # pad image to ensure size is big enough for cropping
        width_pad = max(256 - image.shape[1], 0)
        height_pad = max(256 - image.shape[2], 0)
        image = F.pad(
            image, (0, height_pad, 0, width_pad)
        )  # pylint: disable=not-callable
        target = F.pad(
            target, (0, height_pad, 0, width_pad)
        )  # pylint: disable=not-callable

        # crop image and target
        if self.cropping:
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.crop_size
            )
            image = transforms.functional.crop(image, i, j, h, w)
            target = transforms.functional.crop(target, i, j, h, w)
        else:
            i, j = image.shape[-2:]
            i = i // 2 - 512
            j = j // 2 - 512
            w, h = (1024, 1024)
            image = transforms.functional.crop(image, i, j, h, w)
            target = transforms.functional.crop(target, i, j, h, w)

        # augment image
        if self.augmentations:
            image = self.augmentations(image)

        return image, target.long()

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            length of the dataset

        """
        return len(self.data)
