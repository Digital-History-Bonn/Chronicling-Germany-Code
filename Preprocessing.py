import numpy as np
import torch
from torch.nn.functional import conv2d
from utils import step

SCALE = 4
EXPANSION = 5
THICKEN_ABOVE = 3
THICKEN_UNDER = 0

# TODO: Maybe it's easier to save annotations in one "image"


class Preprocessing:
    def __init__(self, scale=SCALE, expansion=EXPANSION, image_pad_values=(0, 0, 0), target_pad_values=(0, 0, 1),
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
        image = self.scale_img(image)
        image = self._to_numpy(image)
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
        image = self.scale_img(image)
        image = self._to_numpy(image)
        image, mask = self._pad_img(image)

        if target is not None:
            target = self.scale_img(*target)
            target = self._to_numpy(target)
            target, _ = self._pad_img(target)
            target = self._thicken_baseline(target)

            # just use baseline
            target = target[0][None, :]

        return image, target, mask

    def scale_img(self, *args):
        """
        scales down all given images by self.scale
        :param args: images
        :return: list of downscaled images
        """
        size = (args[0].size[0] // self.scale, args[0].size[1] // self.scale)
        for img in args:
            img.thumbnail(size)
        return list(args)

    @staticmethod
    def _to_numpy(image):
        """
        makes image to numpy image and scales colors between 0 and 1
        :param image: image
        :return: np array
        """
        return np.array([np.array(img, dtype=np.float) for img in image]) / 255

    def _pad_img(self, arr, image=True):
        """
        pad image to be dividable by 2^self.expansion
        :param arr: np array of image
        :param image: (bool) if arr is image or target
        :return: padded array and number of pixels added (mask)
        """
        mask = np.zeros(arr.ndim * 2, dtype=int)
        mask[-4:] = (0, self.expansion - (arr.shape[-2] % self.expansion), 0, self.expansion - (arr.shape[-1] % self.expansion))
        mask = mask.reshape(-1, 2)
        arr = np.pad(arr, mask, 'constant', constant_values=self.image_pad_values if image else self.target_pad_values)
        assert arr.shape[1] % 32 == 0 and arr.shape[2] % 32 == 0, f"shape not in the right shape {arr.shape}"
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
    import torch.nn.functional as F
    from Dataloader import Dataloader
    from utils import plot
    pipeline = Preprocessing()

    image = np.ones((3, 5, 5))
    mask = np.zeros(image.ndim * 2, dtype=int)
    mask[-4:] = (0, 32 - (image.shape[-2] % 32), 0, 32 - (image.shape[-1] % 32))
    mask = mask.reshape(-1, 2)

    print(image.shape, mask)
    image = np.pad(image, mask, 'constant', constant_values=((0, 0), (0, 0), (0, 1)))
    print(image)