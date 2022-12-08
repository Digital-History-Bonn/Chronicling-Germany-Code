"""
module for Dataset class
"""
from __future__ import annotations

from typing import Union, Tuple, List
import os
from PIL import Image  # type: ignore
import numpy as np
import torch  # type: ignore
from torch import randperm # type: ignore
from torch.utils.data import Dataset  # type: ignore

from preprocessing import Preprocessing

INPUT = "../prima/inputs/"
TARGETS = "../prima/targets/"


class NewsDataset(Dataset):
    """
        A dataset class for the newspaper datasets
    """

    def __init__(self, paths: List[str] = None, image_path: str = INPUT,
                 target_path: str = TARGETS,
                 limit: Union[None, int] = None, scale: float = None, crop=True):
        """
        Dataset object
        if images and targets are paths to folder with images/np.array
        it loads all images with annotations and preprocesses them
        if images, targets and masks are np.ndarray it just creates a new NewsDataset object

        :param: images (str | list): path to folder of images or list of images
        :param: targets (str | list): path to folder of targets or list of np.arrays
        :param: limit (int): max size of dataloader
        :param scale: scaling of images in preprocessing
        :param crop: if False cropping will be deactivated for prediction purposes. Normalization and training will not
        work on uncropped images.
        """
        self.pipeline = Preprocessing(scale=scale, crop=crop)
        self.crop = crop
        self.scale = scale
        if isinstance(paths, List):
            self.paths = paths

        elif isinstance(image_path, str) and isinstance(target_path, str):

            # load images
            self.paths = [f[:-4] for f in os.listdir(image_path) if f.endswith(".tif")]

            if limit is not None:
                self.paths = self.paths[:limit]

        else:
            raise Exception("Wrong combination of argument types")

    def __len__(self):
        """
        standard len function
        :return: number of items in dateset
        """
        return len(self.paths)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns one datapoint
        :param item: number of the datapoint
        :return (tuple): torch tensor of image, torch tensor of annotation, tuple of mask
        """
        # Open the image form working directory
        file = self.paths[item]
        # load image
        image = Image.open(f"{INPUT}{file}.tif").convert('RGB')

        # load target
        target = np.load(f"{TARGETS}pc-{file}.npy")

        assert image.size[1] == target.shape[0] and image.size[0] == target.shape[1], \
            f"image {file=} has shape {image.size}, but target has shape {target.shape}"

        images, targets = self.pipeline.preprocess(image, target)

        # assert len(images) >= 100, f"number of crops lower than 100 {images.shape=}"

        # indices = randperm(len(images), generator=torch.Generator()).tolist()

        return torch.tensor(images, dtype=torch.float), torch.tensor(targets).long()

    # def class_ratio(self, class_nr: int) -> dict:
    #     """
    #     ratio between the classes over all images
    #     :return: np array of ratio
    #     """
    #     ratio = {c: 0 for c in range(class_nr)}
    #     size = 0
    #     for target in tqdm.tqdm(self.targets, desc='calc class ratio'):
    #         size += target.size
    #         values, counts = np.unique(target, return_counts=True)
    #         for value, count in zip(values, counts):
    #             ratio[value] += count
    #     return {c: v / size for c, v in ratio.items()}

    def random_split(self, ratio: Tuple[float, float, float]) \
            -> Tuple[NewsDataset, NewsDataset, NewsDataset]:
        """
        splits the dataset in parts of size given in ratio
        :param ratio: list[float]:
        :return (list): list of NewsDatasets
        """
        assert sum(ratio) == 1, "ratio does not sum up to 1."
        assert len(ratio) == 3, "ratio does not have length 3"

        splits = int(ratio[0] * len(self)), int(ratio[0] * len(self)) + int(ratio[1] * len(self))

        indices = randperm(len(self), generator=torch.Generator().manual_seed(42)).tolist()
        nd_paths = np.array(self.paths)

        train_dataset = NewsDataset(paths=list(nd_paths[indices[:splits[0]]]), scale=self.scale, crop=self.crop)
        test_dataset = NewsDataset(paths=list(nd_paths[indices[splits[0]:splits[1]]]), scale=self.scale, crop=self.crop)
        valid_dataset = NewsDataset(paths=list(nd_paths[indices[:splits[1]]]), scale=self.scale, crop=self.crop)

        return train_dataset, test_dataset, valid_dataset

    # @property
    # def mean(self) -> torch.Tensor:
    #     """
    #     returns the mean for every color-channel in the dataset
    #     """
    #     # mypy types are ignored to use this class for full image prediction
    #     return torch.tensor(self.images.mean(axis=(0, 2, 3))).float()  # type: ignore
    #
    # @property
    # def std(self) -> torch.Tensor:
    #     """
    #     returns the standard-deviation for every color-channel in the dataset
    #     """
    #     # mypy types are ignored to use this class for full image prediction
    #     return torch.tensor(self.images.std(axis=(0, 2, 3))).float()  # type: ignore

# if __name__ == '__main__':
#     dataset = NewsDataset()
#     train_set, test_set, valid = dataset.random_split((.9, .05, .05))
#     print(f"{type(train_set)=}")
#     print(f"train: {len(train_set)}, test: {len(test_set)}, valid: {len(valid)}")
#     print(f"{train_set.class_ratio(10)}")
#     print()
#     print(train_set.targets[0])
#     print()
#     print(train_set.images[0])
