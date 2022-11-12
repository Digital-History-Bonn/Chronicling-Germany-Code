import os

from typing import List
from PIL import Image
import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import randperm

from Preprocessing import Preprocessing

INPUT = "Data/input/"
TARGETS = "Data/Targets/"


class NewsDataset(Dataset):
    def __init__(self, x: str | list = INPUT, y: str | list = TARGETS, masks=None, limit=None):
        """
        Dataset object
        if x and y are paths to folder with images/np.array it loads all images with annotations and preprocesses them
        if x, y and masks are np.ndarray it just creates a new NewsDataset object
        :param limit: limit for images that be loaded
        """
        if isinstance(x, list) and isinstance(y, list) and isinstance(masks, list):
            self.x = x
            self.y = y
            self.masks = masks

        elif isinstance(x, str) and isinstance(y, str):
            pipeline = Preprocessing()
            self.x, self.y, self.masks = [], [], []

            # load images
            images = [f[:-4] for f in os.listdir(INPUT) if f.endswith(".tif")]

            if limit is not None:
                images = images[:limit]

            # Open the image form working directory
            for file in tqdm.tqdm(images, desc='load data', total=len(images)):

                # load image
                img = Image.open(f"{INPUT}{file}.tif")
                image = Image.new("RGB", img.size)
                image.paste(img)

                # load target
                target = np.load(f"{TARGETS}pc-{file}.npy")

                image, target, mask = pipeline.preprocess(image, target)

                assert image.ndim == 3, f"image does not have right shape({image.shape})"
                assert target.ndim == 2, f"target does not have right shape({target.shape})"
                assert image.shape[1:] == target.shape, "shape of image and target don't match!"

                self.x.append(image)
                self.y.append(target)
                self.masks.append(mask)

        else:
            raise Exception("Wrong combination of argument types")

        assert len(self.x) == len(self.y) == len(self.masks), \
            f"x({len(self.x)}), y({len(self.x)}) and masks ({len(self.x)}) size don't match!"

    def __len__(self):
        """
        standard len function
        :return: number of items in dateset
        """
        return len(self.y)

    def __getitem__(self, item):
        """
        returns one datapoint
        :param item: number of the datapoint
        :return: torch tensor of image, torch tensor of annotation, tuple of mask
        """
        # dummy crop
        images = torch.tensor(self.x[item][:, :512, :512], dtype=torch.float)
        targets = torch.tensor(self.y[item][:512, :512])

        return images, targets, self.masks[item]

    def class_ratio(self, class_nr: int) -> dict:
        """
        ratio between the classes over all images
        :return: np array of ratio
        """
        ratio = {c: 0 for c in range(class_nr)}
        size = 0
        for y in tqdm.tqdm(self.y, desc='calc class ratio'):
            size += y.size
            values, counts = np.unique(y, return_counts=True)
            for v, c in zip(values, counts):
                ratio[v] += c
        return {c: v / size for c, v in ratio.items()}

    def random_split(self, ratio: List[float]):
        """
        splits the dataset in parts of size given in ratio
        :param ratio: list[float]:
        """
        assert sum(ratio) == 1, "ratio does not sum up to 1."
        split = [int(r * len(self)) for r in ratio[1:]]
        split.insert(0, len(self) - sum(split))
        splits = [(sum(split[:x]), sum(split[:x + 1])) for x in range(len(ratio))]

        g = torch.Generator().manual_seed(42)
        indices = randperm(len(self), generator=g).tolist()
        print(f"{indices=}")

        sets = []
        for start, end in splits:
            x, y, masks = [], [], []
            for i in indices[start:end]:
                x.append(self.x[i])
                y.append(self.y[i])
                masks.append(self.masks[i])
            sets.append(NewsDataset(x, y, masks))

        return sets


if __name__ == '__main__':
    dataset = NewsDataset()
    train_set, test_set, valid = dataset.random_split([.9, .05, .05])
    print(f"{type(train_set)=}")
    print(f"train: {len(train_set)}, test: {len(test_set)}, valid: {len(valid)}")
    print(f"{train_set.class_ratio(10)}")
    print(train_set.y[0].shape)
    print(train_set.x[0])
