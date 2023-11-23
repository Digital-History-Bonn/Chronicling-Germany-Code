"""
    module for preprocessing newspaper images and targets
"""

from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from numpy import ndarray
from PIL import Image
from PIL.Image import BICUBIC, NEAREST  # pylint: disable=no-name-in-module
from skimage.util.shape import view_as_windows
from torchvision import transforms

from src.news_seg.utils import correct_shape

SCALE = 1
EXPANSION = 5
THICKEN_ABOVE = 3
THICKEN_UNDER = 0
CROP_FACTOR = 1.5
CROP_SIZE = 512

# REDUCE_CLASSES = {
#     0: [1],
#     4: [2, 3, 5, 6],
#     7: [8]
# }

REDUCE_CLASSES = {
    0: [1],
    7: [8]
}


class Preprocessing:
    """
    class for preprocessing newspaper images and targets
    """

    @staticmethod
    def crop_img(crop_size: int, crop_factor: float, data: ndarray) ->ndarray:
        """
        Crop image by viewing it as windows of size CROP_SIZE x CROP_SIZE and steps of CROP_SIZE // CROP_FACTOR
        :param data: ndarray containing image and target
        :return: ndarray of crops
        """
        windows = np.array(
            view_as_windows(
                data,
                (data.shape[0], crop_size, crop_size),
                step=int(crop_size // crop_factor),
            ),
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

    def __init__(
            self,
            scale: float = SCALE,
            crop_factor: float = CROP_FACTOR,
            crop_size: int = CROP_SIZE,
            crop: bool = True,
            pad: Union[None, Tuple[int, int]] = None,
            reduce_classes: bool = False
    ):
        """
        :param scale: (default: 4)
        :param crop_factor: (default 1.5) step_size of crop is crop_size / crop_factor
        :param crop_size: width and height of crop
        """
        self.scale = scale
        self.crop_factor = crop_factor
        self.crop_size = crop_size
        self.crop = crop
        self.pad = pad
        self.reduce_classes = reduce_classes

    def __call__(
            self, input_image: Image.Image, input_target: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """
        preprocess for image with annotations
        :param image: image
        :param target: annotations
        :return: image, target
        """
        # scale
        image, target = self.scale_img(input_image, input_target)

        self.set_padding(image)

        image, target = self.padding(image, target)
        target = self.replace_labels(target)

        data: npt.NDArray[np.uint8] = np.concatenate(
            (np.array(image, dtype=np.uint8), np.array(target)[np.newaxis, :, :])
        )
        if self.crop:
            data = self.crop_img(self.crop_size, self.crop_factor, data)
            return data

        return np.expand_dims(data, axis=0)

    def set_padding(self, image: torch.Tensor) -> None:
        """
        Set Padding accordingly, if image size is smaller than crop size
        :param image:
        """
        if (image.shape[1] < self.crop_size or image.shape[2] < self.crop_size) and self.crop:
            pad_y = (
                self.crop_size if image.shape[1] < self.crop_size else image.shape[1]
            )
            pad_x = (
                self.crop_size if image.shape[2] < self.crop_size else image.shape[2]
            )
            self.pad = (pad_x, pad_y)
            print(
                f"Image padding because of crop size {self.crop_size} and image shape {image.shape[2]} x "
                f"{image.shape[1]}"
            )

    def replace_labels(self, target: torch.Tensor) -> torch.Tensor:
        """
        Replace labels to reduce classes
        :param target:
        """
        if self.reduce_classes:
            for replace_label, label_list in REDUCE_CLASSES.items():
                for label in label_list:
                    target[target == label] = replace_label
        return target

    def load(
            self, input_path: str, target_path: str, file: str, dataset: str
    ) -> Tuple[Image.Image, ndarray]:
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
        if dataset == "HLNA2013":
            target = target.T

        assert target.dtype == np.uint8
        assert image.size[1] == target.shape[0] and image.size[0] == target.shape[1], (
            f"image {file=} has shape w:{image.size[0]}, h: {image.size[1]}, but target has shape w:{target.shape[1]}, "
            f"h: {target.shape[0]}"
        )

        return image, target

    def padding(
            self, image: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pads border to be divisble by 2**5 to avoid errors during pooling
        :param image:
        :param target:
        :return:
        """
        if self.pad:
            assert self.pad[1] >= image.shape[1] and self.pad[0] >= image.shape[2], (
                f"Final size has to be greater than actual image size. "
                f"Padding to {self.pad[0]} x {self.pad[1]} "
                f"but image has shape of {image.shape[2]} x {image.shape[1]}"
            )

            image = correct_shape(image)
            target = correct_shape(target[None, :])

            transform = transforms.Pad(
                (
                    (self.pad[0] - image.shape[2]) // 2,
                    (self.pad[1] - image.shape[1]) // 2,
                )
            )
            image = transform(image)
            target = transform(target)
            self.pad = None
        return image, torch.squeeze(target)

    def scale_img(
            self, image: Image.Image, target: npt.NDArray[np.uint8]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        scales down all given images and target by scale
        :param image: image
        :param target: target
        return: ndarray tuple containing scaled image and target
        """
        if self.scale == 1:
            image = np.array(image, dtype=np.uint8)
            return torch.tensor(np.transpose(image, (2, 0, 1))), torch.tensor(target)

        shape = int(image.size[0] * self.scale), int(image.size[1] * self.scale)

        image = image.resize(shape, resample=BICUBIC)

        target_img = Image.fromarray(np.array(target.astype(np.uint8)))
        target_img = target_img.resize(shape, resample=NEAREST)

        image, target = np.array(image, dtype=np.uint8), np.array(
            target_img, dtype=np.uint8
        )

        return torch.tensor(np.transpose(image, (2, 0, 1))), torch.tensor(target)


if __name__ == "__main__":
    pass
