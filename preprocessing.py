"""
    module for preprocessing newspaper images and targets
"""

from typing import Tuple

import numpy as np
from PIL import Image  # type: ignore
from numpy import ndarray
from skimage.util.shape import view_as_windows  # type: ignore

SCALE = 0.5
EXPANSION = 5
THICKEN_ABOVE = 3
THICKEN_UNDER = 0
CROP_FACTOR = 1.5
CROP_SIZE = 256


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
    """
    class for preprocessing newspaper images and targets
    """
    def __init__(self, scale=SCALE, expansion: int = EXPANSION,
                 crop_factor: int = CROP_FACTOR, crop_size: int = CROP_SIZE):
        """
        :param scale: (default: 4)
        :param expansion: (default: 5) number of time the image must be scaled to
        :param crop_factor: (default 1.5) step_size of crop is crop_size / crop_factor
        :param crop_size: width and height of crop
        """
        self.scale = scale
        self.expansion = 2 ** expansion
        self.crop_factor = crop_factor
        self.crop_size = crop_size

    def __call__(self, image: Image) -> Image:
        """
        preprocess for the input image only
        :param image: image
        :return: image
        """
        # scale
        t_dummy = np.zeros(image.size)
        image, _ = _scale_img(image, t_dummy)

        return image

    def preprocess(self, image: Image, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        preprocess for image with annotations
        :param image: image
        :param target: annotations
        :return: image, target
        """
        # scale
        image, target = _scale_img(image, target)

        data = self._crop_img(np.concatenate((image, target[np.newaxis, :, :])))

        return data[:, :-1], data[:, -1]

    def _crop_img(self, data: ndarray) -> ndarray:
        """
        Crop image by viewing it as windows of size CROP_SIZE x CROP_SIZE and steps of CROP_SIZE // CROP_FACTOR
        :param data: ndarray containing image and target
        :return: ndarray of crops
        """
        windows = np.array(view_as_windows(data, (data.shape[0], self.crop_size, self.crop_size),
                                           step=int(self.crop_size // self.crop_factor)))
        windows = np.reshape(windows,
                             (np.prod(windows.shape[:3]), windows.shape[3], windows.shape[4], windows.shape[5]))
        return windows


if __name__ == '__main__':
    pass
