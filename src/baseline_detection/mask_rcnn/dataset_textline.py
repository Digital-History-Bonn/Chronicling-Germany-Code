"""Newspaper Class for newspaper mask R-CNN."""

import glob
from typing import Tuple, Dict, Optional

import torch
from skimage import draw, io
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, ToTensor


class CustomDataset(Dataset):  # type: ignore
    """Newspaper Class for training."""
    # TODO: rename Dataset and doc string

    def __init__(self, path: str, transformation: Optional[Module] = None) -> None:
        """
        Newspaper Class for training.

        Args:
            path: path to folder with images
            transformation: torchvision transforms for on-the-fly augmentations
        """
        super().__init__()
        self.data = list(glob.glob(f"{path}/*/*"))
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()
        self.transforms = transformation

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns image and target (boxes, labels, img_number) from dataset.

        Args:
            index: index of datapoint

        Returns:
            image, target

        """
        #TODO: is this returning all polygons from one image? shouldnt those be randomly assambled in a minibatch
        # over multiple images?

        # load image and draw baselines on it
        image = torch.tensor(io.imread(f"{self.data[index]}/image.jpg")).permute(2, 0, 1) / 256
        boxes = torch.load(f"{self.data[index]}/bboxes.pt")

        # load mask polygon targets and create tensors with it
        masks = []
        for polygon in torch.load(f"{self.data[index]}/masks.pt"):
            mask = torch.zeros(image.shape[-2:])
            rr, cc = draw.polygon(polygon[:, 0], polygon[:, 1], image.shape[-2:])
            mask[rr, cc] = 1
            masks.append(mask)

        if self.transforms:
            image = self.transforms(image)

        return (
            image.float(),
            {
                "boxes": torch.stack(boxes),
                "labels": torch.ones(len(masks), dtype=torch.int64),
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
