"""Newspaper Class for newspaper mask R-CNN."""

import glob
from typing import Tuple, Dict, Optional

import torch
from PIL import Image, ImageDraw
from shapely import LineString
from skimage import draw, io
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, ToTensor


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


if __name__ == '__main__':
    from pathlib import Path
    from matplotlib import pyplot as plt
    from src.baseline_detection.mask_rcnn.trainer_textline import draw_prediction

    traindataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../../../data/train_mask"
    )

    image, target = traindataset[3]
    
    print(target['boxes'])

    pred = draw_prediction(image, target)

    plt.imshow(pred.permute(1, 2, 0))
    plt.savefig('imageTest.png', dpi=1000)
