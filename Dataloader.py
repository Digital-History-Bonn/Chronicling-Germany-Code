from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
from tqdm import tqdm
import torch
import torchvision

from Preprocessing import Preprocessing

DOWNSCALE = 3


class Dataloader:
    def __init__(self, folder, limit=None, anno_images=3):
        """
        Dataloader object loads all images with annotations from folder and preprocesses them
        :param folder: path to data
        :param limit: limit for images that be loaded
        :param anno_images: number of images that annotate one image
        """
        self.anno_images = anno_images
        self.pipeline = Preprocessing()
        files = [f for f in listdir(folder) if isfile(join(folder, f)) and f[-4:] == '.JPG']
        files = list(set([f[:9] for f in files]))
        if limit is not None:
            files = files[:limit]

        self.x, self.y, self.masks = [], [], []

        # Open the image form working directory
        for file in tqdm(files, desc='load data'):
            # load image
            image = Image.open(folder + file + '.JPG')

            target = []
            for i in range(self.anno_images):
                channel = Image.open(folder + file + f'_GT{i}' + '.JPG')
                target.append(channel)

            image, target, mask = self.pipeline.preprocess(image, target)

            self.x.append(image)
            self.y.append(target)
            self.masks.append(mask)

        assert len(self.x) == len(self.y) == len(self.masks), f"x, y and masks size don't match!"

    def _sort_by(self, files, first):
        # TODO: delete this function im never used
        dic = {}
        for file in files:
            name = file[:first]
            if name in dic.keys():
                dic[name].append(file)
            else:
                dic[name] = [file]

        return dic

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
        return self.y[0].shape[0]

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
    from utils import plot
    dataloader = Dataloader('data/test/train/', 100)
    print(dataloader.channel)
    print(dataloader.class_ratio)
    X, Y, M = dataloader[0]
    print(X.shape)
    print(Y.shape)

    plot(X[None,:], Y[None,:])
    print(np.max(Y.numpy()))
