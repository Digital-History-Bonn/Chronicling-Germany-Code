"""Newspaper Class for newspaper mask R-CNN."""

import glob
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):  # type: ignore
    """Newspaper Class for training."""
    #TODO: rename Dataset and doc string

    def __init__(self, path: str,
                 scaling: int,
                 augmentations: Optional[Module] = None,
                 cropping: bool = True) -> None:
        """
        Newspaper Class for training.

        Args:
            path: path to folder with images
            scaling: scaling factor for image
            augmentations: torchvision transforms for on-the-fly augmentations
            cropping: whether to crop images randomly during training
        """
        super().__init__()
        self.path = path
        self.scaling = scaling
        self.data = list(glob.glob(f"{path}/*/*"))
        self.cropping = cropping
        self.crop_size = (256, 256)
        self.augmentations = augmentations

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns image and target from dataset.

        Args:
            index: index of datapoint

        Returns:
            image, target

        """
        image = torch.tensor(io.imread(f"{self.data[index]}/image.jpg")).permute(2, 0, 1) / 256
        target = torch.tensor(np.load(f"{self.data[index]}/baselines.npz")['array'])
        target = target.permute(2, 0, 1).float()

        # mask image
        image = image * target[None, 1, :, :]
        target = target[None, 0, :, :]

        # pad image to ensure size is big enough for cropping
        width_pad = max(self.scaling * 256 - image.shape[1], 0)
        height_pad = max(self.scaling * 256 - image.shape[2], 0)
        image = F.pad(image, (0, height_pad, 0, width_pad))     # pylint: disable=not-callable
        target = F.pad(target, (0, height_pad, 0, width_pad))   # pylint: disable=not-callable

        _, width, height = image.shape
        resize = transforms.Resize((width // self.scaling, height // self.scaling))
        image = resize(image)
        target = F.max_pool2d(target, 2)

        # crop image and target
        if self.cropping:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
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
