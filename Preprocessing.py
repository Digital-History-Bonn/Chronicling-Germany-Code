import numpy as np
import torch
from torch.nn.functional import conv2d
from utils import step
from skimage.transform import rescale

SCALE = 1
EXPANSION = 5
THICKEN_ABOVE = 3
THICKEN_UNDER = 0

# TODO: Maybe it's easier to save annotations in one "image"


class Preprocessing:
    def __init__(self, scale=SCALE, expansion=EXPANSION, image_pad_values=(0, 0), target_pad_values=(0, 0, 1),
                 thicken_above=THICKEN_ABOVE, thicken_under=THICKEN_UNDER):
        """

        :param scale: (default: 4)
        :param expansion: (default: 5) number of time the image must be scaled to
        :param image_pad_values: tuple with numbers to pad image with
        :param target_pad_values: (default: (0, 0, 0)) value to pad the different annotation-images with
        :param thicken_above: (default: 5) value to thicken the baselines above
        :param thicken_under: (default: 0) value to thicken the baselines under
        """
        self.scale = scale
        self.expansion = 2**expansion
        self.image_pad_values = tuple([(x, x) for x in image_pad_values])
        self.target_pad_values = tuple([(x, x) for x in target_pad_values])
        self.thicken_above = thicken_above
        self.thicken_under = thicken_under
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
        image = self._scale_img(image)
        image, mask = self._pad_img(image)

        if target is not None:
            target = self._scale_img(target)
            target, _ = self._pad_img(target)

        return image, target, mask

    def _scale_img(self, img: np.array):
        """
        scales down all given images by self.scale
        :param args: images
        :return: list of downscaled images
        """
        if self.scale == 1:
            return img
        # TODO: Fix this! Now it returns not Integer values for classes
        return rescale(img, 1/self.scale, anti_aliasing=False)

    def _pad_img(self, arr: np.array):
        """
        pad image to be dividable by 2^self.expansion
        :param arr: np array of image
        :param image: (bool) if arr is image or target
        :return: padded array and number of pixels added (mask)
        """
        assert arr.ndim == 2, f"{arr.shape=}"
        mask = np.zeros(arr.ndim * 2, dtype=int)
        mask[-4:] = (0, self.expansion - (arr.shape[-2] % self.expansion), 0, self.expansion - (arr.shape[-1] % self.expansion))
        mask = mask.reshape(-1, 2)
        arr = np.pad(arr, mask, 'constant', constant_values=0)              # maybe make constant_values a variable

        assert arr.shape[0] % 32 == 0 and arr.shape[1] % 32 == 0, f"shape not in the right shape {arr.shape}"

        return arr, mask

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
