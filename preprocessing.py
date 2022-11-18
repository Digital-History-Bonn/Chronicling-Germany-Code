from typing import Tuple

import numpy as np
import torch
from PIL import Image  # type: ignore
from numpy import ndarray
from torchvision import transforms  # type: ignore


SCALE = 0.5
EXPANSION = 5
THICKEN_ABOVE = 3
THICKEN_UNDER = 0
CROP_FACTOR = 2
CROP_SIZE = 256


# TODO: Maybe it's easier to save annotations in one "image"


def get_max_crops(image_size: Tuple[int, int], crop_size: Tuple[int, int]) -> int:
    """
    calculates number of cropped images that can be extracted without much overlapping.
    :param image_size: image size
    :param crop_size: size of cropped image
    """
    return int((image_size[0] / crop_size[0]) * (image_size[1] / crop_size[1]))


def _scale_img(image: Image, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    scales down all given image and target by scale
    :param image (Image): image
    :param target (np.ndarray): target
    return: ndarray tuple containing scaled image and target
    """
    if SCALE == 1:
        image = np.array(image)
        return np.transpose(image, (2, 0, 1)), target

    shape = int(image.size[0] * SCALE), int(image.size[1] * SCALE)

    image = image.resize(shape, resample=Image.BICUBIC)

    target_img = Image.fromarray(target.astype(np.uint8))
    target_img = target_img.resize(shape, resample=Image.NEAREST)

    image, target = np.array(image), np.array(target_img)  # type: ignore

    return np.transpose(image, (2, 0, 1)), target


class Preprocessing:
    def __init__(self, scale=SCALE, expansion=EXPANSION, image_pad_values=(0, 0), target_pad_values=(0, 0, 1),
                 crop_factor=CROP_FACTOR, crop_size=CROP_SIZE):
        """
        :param scale: (default: 4)
        :param expansion: (default: 5) number of time the image must be scaled to
        :param image_pad_values: tuple with numbers to pad image with
        :param target_pad_values: (default: (0, 0, 0)) value to pad the different annotation-images with
        """
        self.scale = scale
        self.expansion = 2 ** expansion
        self.image_pad_values = tuple([(x, x) for x in image_pad_values])
        self.target_pad_values = tuple([(x, x) for x in target_pad_values])
        self.crop_factor = crop_factor
        self.crop_size = crop_size

    def __call__(self, image: Image):
        """
        preprocess for the input image only
        :param image: image
        :return: image
        """
        # scale
        t_dummy = np.zeros(image.size)
        image, _ = _scale_img(image, t_dummy)

        return image

    def preprocess(self, image: Image, target: np.ndarray):
        """
        preprocess for image with annotations
        :param image: image
        :param target: annotations
        :return: image, target
        """
        # scale
        image, target = _scale_img(image, target)

        count = get_max_crops(image.shape[-2:], (self.crop_size, self.crop_size))
        data = self._crop_img(np.concatenate((image, target[np.newaxis, :, :])), count)

        return data[:, :-1], data[:, -1]

    def _crop_img(self, data: ndarray, count: int) -> ndarray:
        """
        random crop image and targets.
        :param data: ndarray containing image and target
        :param count: count of to be cropped images
        """
        count = int(count * self.crop_factor)
        if count == 0:
            return data

        data = torch.tensor(data)  # type: ignore
        transform = transforms.RandomCrop((self.crop_size, self.crop_size))

        images = []
        for _ in range(count):
            images.append(np.array(transform(data)))
        return np.array(images)


if __name__ == '__main__':
    pass
