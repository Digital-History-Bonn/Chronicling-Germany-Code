from typing import Tuple

import numpy as np
import torch
import PIL
from numpy import ndarray
from torch.nn.functional import conv2d
from torchvision import transforms

from utils import step

SCALE = 1
EXPANSION = 5
THICKEN_ABOVE = 3
THICKEN_UNDER = 0
CROP_FACTOR = 2
CROP_SIZE = 512


# TODO: Maybe it's easier to save annotations in one "image"


def get_max_crops(image_size: Tuple[int, int], crop_size: Tuple[int, int]) -> int:
    """
    calculates number of cropped images that can be extracted without much overlapping.
    :param image_size: image size
    :param crop_size: size of cropped image
    """
    return int((image_size[0] / crop_size[0]) * (image_size[1] / crop_size[1]))


class Preprocessing:
    def __init__(self, scale=SCALE, expansion=EXPANSION, image_pad_values=(0, 0), target_pad_values=(0, 0, 1),
                 thicken_above=THICKEN_ABOVE, thicken_under=THICKEN_UNDER, crop_factor=CROP_FACTOR,
                 crop_size=CROP_SIZE):
        """

        :param scale: (default: 4)
        :param expansion: (default: 5) number of time the image must be scaled to
        :param image_pad_values: tuple with numbers to pad image with
        :param target_pad_values: (default: (0, 0, 0)) value to pad the different annotation-images with
        :param thicken_above: (default: 5) value to thicken the baselines above
        :param thicken_under: (default: 0) value to thicken the baselines under
        """
        self.scale = scale
        self.expansion = 2 ** expansion
        self.image_pad_values = tuple([(x, x) for x in image_pad_values])
        self.target_pad_values = tuple([(x, x) for x in target_pad_values])
        self.thicken_above = thicken_above
        self.thicken_under = thicken_under
        self.crop_factor = crop_factor
        self.crop_size = crop_size
        self.weights_size = 2 * (max(thicken_under, thicken_above))

    def __call__(self, image):
        """
        preprocess for the input image only
        :param image: image
        :return: image, mask
        """
        # scale
        image = self._scale_img(image)
        image, mask = self._pad_img(image)

        return image, mask

    def preprocess(self, image, target):
        """
        preprocess for image with annotations
        :param image: image
        :param target: annotations
        :return: image, target, mask
        """
        # scale
        image = self._scale_img(image, 255)
        image, mask = self._pad_img(image)

        if target is not None:
            # TODO: spacing is dependent on label count.
            target = self._scale_img(target, 1)
            target, _ = self._pad_img(target)

        count = get_max_crops(image.shape, (self.crop_size, self.crop_size))
        data = self._crop_img(np.array([image, target]), count)

        return data[0], data[1], mask

    def _scale_img(self, image: ndarray, spacing):
        """
        scales down all given images by self.scale
        :param image: image
        :return: list of downscaled images
        """
        image = image * spacing
        pil_img = PIL.Image.fromarray(image.astype('int8'))
        if self.scale == 1:
            return np.asarray(pil_img, dtype=np.float32) / spacing
        width, height = int(SCALE * pil_img.size[0]), int(SCALE * pil_img.size[1])
        pil_img = pil_img.resize((width, height), resample=PIL.Image.NEAREST)
        return np.asarray(pil_img, dtype=np.float32) / spacing

    def _pad_img(self, arr: ndarray):
        """
        pad image to be dividable by 2^self.expansion
        :param arr: np array of image
        :param image: (bool) if arr is image or target
        :return: padded array and number of pixels added (mask)
        """
        assert arr.ndim == 2, f"{arr.shape=}"
        mask = np.zeros(arr.ndim * 2, dtype=int)
        mask[-4:] = (
            0, self.expansion - (arr.shape[-2] % self.expansion), 0, self.expansion - (arr.shape[-1] % self.expansion))
        mask = mask.reshape(-1, 2)
        arr = np.pad(arr, mask, 'constant', constant_values=0)  # maybe make constant_values a variable

        assert arr.shape[0] % 32 == 0 and arr.shape[1] % 32 == 0, f"shape not in the right shape {arr.shape}"

        return arr, mask

    def _crop_img(self, data: ndarray, count: int) -> ndarray:
        """
        random crop image and targets.
        :param data: ndarray containing image and target
        :param count: count of to be cropped images
        """
        count = count * self.crop_factor
        transform = transforms.RandomCrop((self.crop_size, self.crop_size))

        images = []
        for _ in range(count):
            images.append(np.array(transform(torch.tensor(data))))
        return np.transpose(np.array(images), (1, 0, 2, 3))

    def _thicken_baseline(self, target, dim=0):
        target[dim] = self._thicken(target[dim])
        return target

    def _thicken(self, image):
        image = torch.tensor(image[None, :])
        weights = np.zeros(self.weights_size)
        weights[(self.weights_size // 2) - self.thicken_under:(self.weights_size // 2) + self.thicken_above] = 1
        weights = torch.tensor(weights[None, None, :, None])
        image = conv2d(image, weights, stride=1, padding='same')[0].numpy()
        return step(image)


if __name__ == '__main__':
    pass
