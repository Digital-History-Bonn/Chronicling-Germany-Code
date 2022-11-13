"""
module for Dataset class
"""
from typing import Union

import os
from PIL import Image
import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import randperm

from preprocessing import Preprocessing

INPUT = "Data/input/"
TARGETS = "Data/Targets/"


class NewsDataset(Dataset):
    """
        A dataset class for the newspaper datasets
    """
    def __init__(self, images: Union[str, list] = INPUT,
                 targets: Union[str, list] = TARGETS,
                 limit: Union[None, int] = None):
        """
        Dataset object
        if images and targets are paths to folder with images/np.array
        it loads all images with annotations and preprocesses them
        if images, targets and masks are np.ndarray it just creates a new NewsDataset object

        :param: images (str | list): path to folder of images or list of images
        :param: targets (str | list): path to folder of targets or list of np.arrays
        :param: limit (int): max size of dataloader
        """
        if isinstance(images, list) and isinstance(targets, list):
            self.images = images
            self.targets = targets

        elif isinstance(images, str) and isinstance(targets, str):
            pipeline = Preprocessing()
            self.images, self.targets = [], []

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

                images, targets = pipeline.preprocess(image, target)

                self.images.extend(images)
                self.targets.extend(targets)

        else:
            raise Exception("Wrong combination of argument types")

    def __len__(self):
        """
        standard len function
        :return: number of items in dateset
        """
        return len(self.targets)

    def __getitem__(self, item: int) -> tuple[torch.tensor, torch.tensor]:
        """
        returns one datapoint
        :param item: number of the datapoint
        :return (tuple): torch tensor of image, torch tensor of annotation, tuple of mask
        """
        return torch.tensor(self.images[item], dtype=torch.float),\
               torch.tensor(self.targets[item])

    def class_ratio(self, class_nr: int) -> dict:
        """
        ratio between the classes over all images
        :return: np array of ratio
        """
        ratio = {c: 0 for c in range(class_nr)}
        size = 0
        for target in tqdm.tqdm(self.targets, desc='calc class ratio'):
            size += target.size
            values, counts = np.unique(target, return_counts=True)
            for value, count in zip(values, counts):
                ratio[value] += count
        return {c: v / size for c, v in ratio.items()}

    def random_split(self, ratio: list[float]):
        """
        splits the dataset in parts of size given in ratio
        :param ratio: list[float]:
        :return (list): list of NewsDatasets
        """
        assert sum(ratio) == 1, "ratio does not sum up to 1."
        split = [int(r * len(self)) for r in ratio[1:]]
        split.insert(0, len(self) - sum(split))
        splits = [(sum(split[:x]), sum(split[:x + 1])) for x in range(len(ratio))]

        indices = randperm(len(self), generator=torch.Generator().manual_seed(42)).tolist()
        print(f"{indices=}")

        sets = []
        for start, end in splits:
            img, tar = [], []
            for i in indices[start:end]:
                img.append(self.images[i])
                tar.append(self.targets[i])
            sets.append(NewsDataset(img, tar))

        return sets


if __name__ == '__main__':
    dataset = NewsDataset()
    train_set, test_set, valid = dataset.random_split([.9, .05, .05])
    print(f"{type(train_set)=}")
    print(f"train: {len(train_set)}, test: {len(test_set)}, valid: {len(valid)}")
    print(f"{train_set.class_ratio(10)}")
    print(train_set.targets[0].shape)
    print(train_set.images[0])
