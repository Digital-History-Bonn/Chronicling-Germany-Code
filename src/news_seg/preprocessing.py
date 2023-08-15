"""
    module for preprocessing newspaper images and targets
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image
from PIL.Image import BICUBIC, NEAREST # pylint: disable=no-name-in-module
from numpy import ndarray
from skimage.util.shape import view_as_windows

SCALE = 1
EXPANSION = 5
THICKEN_ABOVE = 3
THICKEN_UNDER = 0
CROP_FACTOR = 1.5
CROP_SIZE = 256


class Preprocessing:
    """
    class for preprocessing newspaper images and targets
    """

    def __init__(
        self,
        scale: float = SCALE,
        expansion: int = EXPANSION,
        crop_factor: float = CROP_FACTOR,
        crop_size: int = CROP_SIZE,
        crop: bool = True,
    ):
        """
        :param scale: (default: 4)
        :param expansion: (default: 5) number of time the image must be scaled to
        :param crop_factor: (default 1.5) step_size of crop is crop_size / crop_factor
        :param crop_size: width and height of crop
        """
        self.scale = scale
        self.expansion = 2**expansion
        self.crop_factor = crop_factor
        self.crop_size = crop_size
        self.crop = crop

    def __call__(
        self, image: Image, target: npt.NDArray[np.uint]
    ) -> Tuple[npt.NDArray[np.single], npt.NDArray[np.uint]]:
        """
        preprocess for image with annotations
        :param image: image
        :param target: annotations
        :return: image, target
        """
        # scale
        image, target = self.scale_img(image, target)

        if self.crop:
            data = self.crop_img(np.concatenate((image, target[np.newaxis, :, :])))
            return data[:, :-1], data[:, -1]
        return image, target

    def load(self, input_path: str, target_path: str, file: str) -> Tuple[Image, ndarray]:
        """Load image and target
        :param input_path: path to input image
        :param target_path: path to target
        :param file: name of image and target
        :return:
        """
        # load image
        image = Image.open(f"{input_path}").convert("RGB")

        # load target
        target = np.load(f"{target_path}")

        assert (
            image.size[1] == target.shape[0] and image.size[0] == target.shape[1]
        ), f"image {file=} has shape {image.size}, but target has shape {target.shape}"

        return image, target

    def scale_img(
        self, image: Image, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        scales down all given images and target by scale
        :param image (Image): image
        :param target (np.ndarray): target
        return: ndarray tuple containing scaled image and target
        """
        if self.scale == 1:
            image = np.array(image)
            return np.transpose(image, (2, 0, 1)), target

        shape = int(image.size[0] * self.scale), int(image.size[1] * self.scale)

        image = image.resize(shape, resample=BICUBIC)

        target_img = Image.fromarray(target.astype(np.uint8))
        target_img = target_img.resize(shape, resample=NEAREST)

        image, target = np.array(image), np.array(target_img)

        return np.transpose(image, (2, 0, 1)), target

    def crop_img(self, data: ndarray) -> ndarray:
        """
        Crop image by viewing it as windows of size CROP_SIZE x CROP_SIZE and steps of CROP_SIZE // CROP_FACTOR
        :param data: ndarray containing image and target
        :return: ndarray of crops
        """
        windows = np.array(
            view_as_windows(
                data,
                (data.shape[0], self.crop_size, self.crop_size),
                step=int(self.crop_size // self.crop_factor),
            )
        )
        windows = np.reshape(
            windows,
            (
                np.prod(windows.shape[:3]),
                windows.shape[3],
                windows.shape[4],
                windows.shape[5],
            ),
        )
        return windows


if __name__ == "__main__":
    pass
