from Preprocessing import Preprocessing

from skimage import io
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import os


DOWNSCALE = 3
INPUT = "Data/input/"
TARGETS = "Data/Targets/"


# TODO: Random_split implementieren


class Dataset(Dataset):
    def __init__(self, limit=None):
        """
        Dataloader object loads all images with annotations from folder and preprocesses them
        :param limit: limit for images that be loaded
        """
        self.pipeline = Preprocessing()
        self.x, self.y, self.masks = [], [], []

        # load images
        images = [f for f in os.listdir(INPUT) if f.endswith(".tif")]

        # load targets
        targets = [f for f in os.listdir(TARGETS) if f.endswith(".npy")]

        if limit is not None:
            targets = targets[:limit]
            images = images[:limit]

        # Open the image form working directory
        print(len(images), len(targets))
        for image_path, target_path in tqdm(zip(images, targets), desc='load data', total=len(images)):
            # load image
            image = io.imread(f"{INPUT}{image_path}", as_gray=True)
            target = np.load(f"{TARGETS}{target_path}")

            image, target, mask = self.pipeline.preprocess(image, target)

            self.x.append(image)
            self.y.append(target)
            self.masks.append(mask)

        assert len(self.x) == len(self.y) == len(self.masks), f"x, y and masks size don't match!"

    def sample(self, nr):
        """
        returns a sample of the dataset
        :param nr: number of images in the sample
        :return: list with samples
        """
        data = []
        for i in np.random.randint(0, self.__len__(), nr):
            data.append(self.__getitem__(i))

        return data

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
        return torch.tensor(np.array(self.x[item]), dtype=torch.float), torch.tensor(np.array(self.y[item]),
                                                                                       dtype=torch.int64), self.masks[
                   item]

    @property
    def channel(self):
        """
        :return: number of predicted classes
        """
        return self.y

    @property
    def class_ratio(self):
        """
        ratio between the classes over all images
        :return: np array of ratio
        """
        ratio = np.zeros(self.channel)
        add = np.sum(np.array([np.unique(y, return_counts=True)[1] for y in self.y]), axis=0)
        return (ratio + add) / np.sum(add)


if __name__ == '__main__':
    dataset = Dataset()
    print(len(dataset))