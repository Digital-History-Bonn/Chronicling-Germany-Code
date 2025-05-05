"""
module for preprocessing newspaper images and targets
"""

from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from numpy import ndarray
from PIL import Image
from PIL.Image import (  # pylint: disable=no-name-in-module # type: ignore
    BICUBIC,
    NEAREST,
)
from skimage.util.shape import view_as_windows
from torchvision import transforms

from src.cgprocess.layout_segmentation.utils import replace_labels

SCALE = 1
EXPANSION = 5
THICKEN_ABOVE = 3
THICKEN_UNDER = 0
CROP_FACTOR = 1.5
CROP_SIZE = 512


class Preprocessing:
    """
    class for preprocessing newspaper images and targets
    """

    @staticmethod
    def calculate_padding_size(
        image: torch.Tensor, size: int, factor: float
    ) -> Tuple[int, int]:
        """
        Sets padding to make the image compatible with cropping. For this, it can not be smaller than one crop
        at each dimension. If a dimension is of greater size than a crop, it will be padded to be a multiple of
        the crop step size. This prevents the last crop to the right and bottom to be dropped.
        :param image: image tensor with [..., C, H, W]
        """
        shape = (image.shape[-1], image.shape[-2])
        crop_step = int(size // factor)
        if shape[0] < size:
            pad_x = size - shape[0]
        elif shape[0] % crop_step > 0:
            pad_x = crop_step - (shape[0] % crop_step)
        else:
            pad_x = 0

        if shape[1] < size:
            pad_y = size - shape[1]
        elif shape[1] % crop_step > 0:
            pad_y = crop_step - (shape[1] % crop_step)
        else:
            pad_y = 0

        return pad_x, pad_y

    @staticmethod
    def crop_img(crop_size: int, crop_factor: float, data: ndarray) -> ndarray:
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
        reduce_classes: bool = False,
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
        self.pad: Union[Tuple[int, int], None] = None
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
        target = replace_labels(target) if self.reduce_classes else target

        data: npt.NDArray[np.uint8] = np.concatenate(
            (np.array(image, dtype=np.uint8), np.array(target)[np.newaxis, :, :])
        )
        if self.crop:
            data = self.crop_img(self.crop_size, self.crop_factor, data)
            return data

        return np.expand_dims(data, axis=0)

    def set_padding(self, image: torch.Tensor) -> None:
        """
        Sets padding to make the image compatible with cropping. For this, it can not be smaller than one crop
        at each dimension. If a dimension is of greater size than a crop, it will be padded to be a multiple of
        the crop step size. This prevents the last crop to the right and bottom to be dropped.
        """
        if self.crop:
            self.pad = self.calculate_padding_size(
                image, self.crop_size, self.crop_factor
            )

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
        target = np.load(f"{target_path}")["array"]
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
        Pads image by given size to the right and bottom.
        """
        if self.pad:
            transform = transforms.Pad(
                (
                    0,
                    0,
                    self.pad[0],
                    self.pad[1],
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
            image_ndarray = np.array(image, dtype=np.uint8)
            return torch.tensor(np.transpose(image_ndarray, (2, 0, 1))), torch.tensor(
                target
            )

        shape = int(image.size[0] * self.scale), int(image.size[1] * self.scale)

        image = image.resize(shape, resample=BICUBIC)

        target_img = Image.fromarray(np.array(target.astype(np.uint8)))
        target_img = target_img.resize(shape, resample=NEAREST)

        image_ndarray, target = np.array(image, dtype=np.uint8), np.array(
            target_img, dtype=np.uint8
        )

        return torch.tensor(np.transpose(image_ndarray, (2, 0, 1))), torch.tensor(
            target
        )


if __name__ == "__main__":
    pass
