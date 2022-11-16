"""
    module for preprocessing newspaper images and target
"""

import numpy as np
from PIL import Image
from PIL.Image import NEAREST
from numpy import ndarray

SCALE = 1
EXPANSION = 5
THICKEN_ABOVE = 3
THICKEN_UNDER = 0


class Preprocessing:
    """
        class for preprocessing newspaper images and target
    """
    def __init__(self, scale: float = SCALE,
                 expansion: int = EXPANSION,
                 image_pad_value: int = 0,
                 target_pad_value: int = 0):
        """
        class for preprocessing newspaper images and target

        :param scale (float): value of down scaling image (default: 1)
        :param expansion (int): number of time the image must be scaled to (default: 5)
        :param image_pad_value (int): value for padding image (default: 0)
        :param target_pad_value (int): value for padding target (default: 0)
        """
        self.scale = scale
        self.expansion = 2 ** expansion
        self.image_pad_value = image_pad_value
        self.target_pad_value = target_pad_value

    def __call__(self, image: Image):
        """
        preprocess for the input image only
        :param image (Image): image
        :return: image, mask
        """
        # scale
        t_dummy = np.zeros(image.size)
        image, _ = self._scale_img(image, t_dummy)
        image, mask = self._pad_img(image)

        return image, mask

    def preprocess(self, image: Image, target: np.ndarray):
        """
        preprocess for image with annotations
        :param image: image
        :param target: annotations
        :return: image, target, mask
        """
        # scale
        image, target = self._scale_img(image, target)
        images, targets = self._dummy_crop(image, target)

        return images, targets

    @staticmethod
    def _scale_img(image: Image, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        scales down all given image and target by scale
        :param image (Image): image
        :param target (np.ndarray): target
        """
        if SCALE == 1:
            image = np.array(image)
            return np.transpose(image, (2, 0, 1)), target

        shape = int(image.size[0] * SCALE), int(image.size[1] * SCALE)

        image = image.resize(shape, resample=Image.BICUBIC)

        target = Image.fromarray(target.astype(np.uint8))
        target = target.resize(shape, resample=Image.BICUBIC)

        image, target = np.array(image), np.array(target)

        return np.transpose(image, (2, 0, 1)), target

    @staticmethod
    def _dummy_crop(image: np.ndarray, target: np.ndarray) \
            -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
            simple cropping function to be replaced later
        """
        return [image[:, :512, :512], image[:, -512:, -512:]], \
               [target[:512, :512], target[-512:, -512:]]


if __name__ == '__main__':
    pass
