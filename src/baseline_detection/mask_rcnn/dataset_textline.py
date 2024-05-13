"""Newspaper Class for newspaper mask R-CNN."""

import glob
from typing import Tuple, Dict, Optional

import torch
from skimage import draw, io
from torch.nn import Module
from torch.utils.data import Dataset


class CustomDataset(Dataset):  # type: ignore
    """Newspaper Class for training."""

    def __init__(self, path: str, transformation: Optional[Module] = None) -> None:
        """
        Newspaper Class for training.

        Args:
            path: path to folder with images
            transformation: torchvision transforms for on-the-fly augmentations
        """
        super().__init__()
        self.data = [x for x in glob.glob(f"{path}/*/*")]
        self.transforms = transformation

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns image and target (boxes, labels, img_number) from dataset.

        Args:
            index: index of datapoint

        Returns:
            image, target

        """
        # load image and targets depending on objective
        image = torch.tensor(io.imread(f"{self.data[index]}/image.jpg")).permute(2, 0, 1)
        boxes = torch.load(f"{self.data[index]}/bboxes.pt")
        polygons = torch.load(f"{self.data[index]}/masks.pt")
        masks = []
        for polygon in polygons:
            mask = torch.zeros(image.shape[-2:])
            rr, cc = draw.polygon(polygon[:, 0], polygon[:, 1], image.shape[-2:])
            mask[rr, cc] = 1
            masks.append(mask)

        if self.transforms:
            image = self.transforms(image)

        return (
            image.float(),
            {
                "boxes": boxes,
                "labels": torch.ones(len(boxes), dtype=torch.int64),
                "masks": torch.stack(masks),
            },
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            length of the dataset

        """
        return len(self.data)
