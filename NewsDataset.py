from typing import List

from Preprocessing import Preprocessing

from skimage import io
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch import randperm
import os

INPUT = "Data/scale4/input/"
TARGETS = "Data/scale4/Targets/"


class NewsDataset(Dataset):
    def __init__(self, x=INPUT, y=TARGETS, masks=None, limit=None):
        """
        Dataset object
        if x and y are paths to folder with images/np.array it loads all images with annotations and preprocesses them
        if x, y and masks are np.ndarray it just creats a new NewsDataset object
        :param limit: limit for images that be loaded
        """
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(masks, np.ndarray):
            self.x = np.array(x)
            self.y = np.array(y)
            self.masks = np.array(masks)

        elif isinstance(x, str) and isinstance(y, str):
            pipeline = Preprocessing()
            self.x, self.y, self.masks = [], [], []

            # load images
            images = [f[:-4] for f in os.listdir(INPUT) if f.endswith(".tif")]

            if limit is not None:
                images = images[:limit]

            # Open the image form working directory
            for file in tqdm(images, desc='load data', total=len(images)):
                # load image
                image = io.imread(f"{INPUT}{file}.tif", as_gray=True)
                target = np.load(f"{TARGETS}pc-{file}.npy")

                image, target, mask = pipeline.preprocess(image, target)

                self.x.append(image)
                self.y.append(target)
                self.masks.append(mask)

            self.x = np.array(self.x)
            self.y = np.array(self.y)
            self.masks = np.array(self.masks)

        else:
            raise Exception("Wrong combination of argument types")

        assert len(self.x) == len(self.y) == len(self.masks), f"x, y and masks size don't match!"

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
        return torch.tensor(np.array([self.x[item]]), dtype=torch.float), torch.tensor(np.array(self.y[item]),
                                                                                       dtype=torch.int64), self.masks[
                   item]

    def save(self, path):
        np.savez_compressed(path, x=self.x, y=self.y, masks=self.masks)

    @classmethod
    def load(cls, path):
        loaded = np.load(path)
        return Dataset(loaded['x'], loaded['y'], loaded['masks'])

    def class_ratio(self, class_nr: int):
        """
        ratio between the classes over all images
        :return: np array of ratio
        """
        ratio = {c: 0 for c in range(class_nr)}
        size = 0
        for y in tqdm(self.y, desc='calc class ratio'):
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
        splits = [int(r * len(self)) for r in ratio[1:]]
        splits.insert(0, len(self) - sum(splits))
        splits = [(sum(splits[:x]), sum(splits[:x + 1])) for x in range(len(ratio))]

        g = torch.Generator().manual_seed(42)
        indices = randperm(len(self), generator=g).tolist()

        return [NewsDataset(self.x[indices[x:y]], self.y[indices[x:y]], self.masks[indices[x:y]]) for x, y in splits]


if __name__ == '__main__':

    dataset = NewsDataset()
    dataset.save('Data/datasets/text')
    dataset = NewsDataset.load('Data/datasets/text.npz')
    train_set, test_set, valid = dataset.random_split([.9, .05, .05])
    print(f"{type(train_set)=}")
    print(f"train: {len(train_set)}, test: {len(test_set)}, valid: {len(valid)}")
