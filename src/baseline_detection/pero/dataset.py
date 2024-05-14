"""Newspaper Class for newspaper mask R-CNN."""

import os
import glob
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision import transforms


class RandomCropAndResize(Module):
    """Crops the given image and target at a random position."""

    def __init__(self, size: Tuple[int, int]):
        """
        Crops the given image and target at a random position.

        Args:
            size: Size of the crop
        """
        super().__init__()
        self.size = size

    def __call__(self, image: torch.Tensor, target: torch.Tensor):
        """
        Crops the given image and target at a random position.

        Args:
            image: torch Tensor representation of the image (channel, width, height)
            target: torch Tensor representation of the target (channel, width, height)

        Returns:
            cropped image: torch Tensor representation of the image (channel, size[0], size[1])
            cropped target torch Tensor representation of the target (channel, size[0], size[1])
        """
        # Randomly crop the image and mask
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.size)
        image = transforms.functional.crop(image, i, j, h, w)
        target = transforms.functional.crop(target, i, j, h, w)
        return image, target


class CustomDataset(Dataset):  # type: ignore
    """Newspaper Class for training."""

    def __init__(self, image_path: str,
                 target_path: str,
                 augmentations: Module = None,
                 cropping: bool = True) -> None:
        """
        Newspaper Class for training.

        Args:
            image_path: path to folder with images
            target_path: path to folder with targets
            augmentations: torchvision transforms for on-the-fly augmentations
            cropping: whether to crop images or not
        """
        super().__init__()
        self.image_path = image_path
        self.target_path = target_path
        self.data = [x.split(os.sep)[-1][:-4] for x in glob.glob(f"{target_path}/*")]
        self.cropping = cropping
        self.crop = RandomCropAndResize(size=(256, 256))
        self.augmentations = augmentations

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns image and target (boxes, labels, img_number) from dataset.

        Args:
            index: index of datapoint

        Returns:
            image, target
        """
        image = torch.tensor(io.imread(f"{self.image_path}/{self.data[index]}.jpg"))
        image = image.permute(2, 0, 1) / 256
        target = torch.tensor(np.load(f"{self.target_path}/{self.data[index]}.npz")['array'])
        target = target.permute(2, 0, 1).float()

        # mask image
        image = image * target[None, 5, :, :]
        target = target[:4, :, :]

        # pad image to ensure size is big enough for cropping
        width_pad = max(256 - image.shape[1], 0)
        height_pad = max(256 - image.shape[2], 0)
        image = F.pad(image, (0, height_pad, 0, width_pad))
        target = F.pad(target, (0, height_pad, 0, width_pad))

        _, width, height = image.shape
        resize = transforms.Resize((width // 2, height // 2))
        image = resize(image)
        target = F.max_pool2d(target, 2)

        # crop image and target
        if self.cropping:
            image, target = self.crop(image, target)
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
