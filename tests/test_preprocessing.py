"""Test class for preprocessing"""

import numpy as np

# pylint: disable-next=no-name-in-module
# pylint: disable-next=import-error
import pytest
import torch
from PIL import Image

from src.cgprocess.layout_segmentation.processing.preprocessing import Preprocessing

DATA_PATH = "./tests/data/preprocessing/"


class TestClassPreprocessing:
    """Class for testing preprocessing"""

    @pytest.fixture(autouse=True)
    def setup(self):
        "will initiate NewsDataset for every test"
        np.random.seed(314)
        pytest.preprocessing = Preprocessing()

    def test_load(self):
        """verify image and target loading
        note that targets and images have opposing order of dimension wich further preprocessing steps handle
        """
        image, target = pytest.preprocessing.load(
            f"{DATA_PATH}input/test-image.jpg",
            f"{DATA_PATH}input/test-target.npz",
            "test",
            dataset="transcibus",
        )
        assert image.mode == "RGB"
        assert f"{target.dtype}" == "uint8"

    def test_call(self):
        """Verify entire preprocessing"""

        # not matching crop test
        size = 100
        channels = 3
        crop_size = 512
        pytest.preprocessing.crop_size = crop_size
        pytest.preprocessing.crop_factor = 1
        image = Image.fromarray(
            (np.random.rand(size, size, channels) * 255).astype("uint8")
        ).convert("RGB")
        target = np.random.randint(0, 10, (size, size), dtype=np.uint8)

        result_data = pytest.preprocessing(image, target)

        assert result_data.shape == (
            1,
            channels + 1,
            crop_size,
            crop_size, # Padding is expanded, such that the image can pass the model with eight downscaling operations
        )
        assert result_data.dtype == np.uint8

        # standart crop test
        size = 100
        channels = 3
        crop_size = 1024
        pytest.preprocessing.crop_size = crop_size
        pytest.preprocessing.crop_factor = 1
        image = Image.fromarray(
            (np.random.rand(size, size, channels) * 255).astype("uint8")
        ).convert("RGB")
        target = np.random.randint(0, 10, (size, size), dtype=np.uint8)

        result_data = pytest.preprocessing(image, target)

        assert result_data.shape == (
            1,
            channels + 1,
            crop_size,
            crop_size,
        )
        assert result_data.dtype == np.uint8

        # oversized crop test
        crop_size = 256
        pytest.preprocessing.crop_size = crop_size
        result_data = pytest.preprocessing(image, target)

        assert result_data.shape == (
            1,
            channels + 1,
            crop_size,
            crop_size,
        )
        assert result_data.dtype == np.uint8

        # padding test
        pytest.preprocessing.crop = False
        pad_size = 156
        pytest.preprocessing.pad = pad_size, pad_size

        result_data = pytest.preprocessing(image, target)
        assert result_data.shape == (1, channels + 1, size + pad_size, size + pad_size)
        assert result_data.dtype == np.uint8

        # uneven size padding test
        pytest.preprocessing.crop = False
        pad_size = 156
        pytest.preprocessing.pad = pad_size, pad_size
        size = 101
        image = Image.fromarray(
            (np.random.rand(size, size, channels) * 255).astype("uint8")
        ).convert("RGB")
        target = np.random.randint(0, 10, (size, size), dtype=np.uint8)

        result_data = pytest.preprocessing(image, target)
        assert result_data.shape == (1, channels + 1, 512, 512)
        assert result_data.dtype == np.uint8

    def test_set_padding(self):
        """verify pading setting if crop is too large"""
        # standart test
        size_x = 100
        size_y = 100
        channels = 3
        crop_size = 150
        pytest.preprocessing.crop_size = crop_size
        pytest.preprocessing.crop_factor = 1
        image = torch.Tensor(
            (np.random.rand(channels, size_y, size_x) * 255).astype("uint8")
        )

        # y greater than cropsize test
        pytest.preprocessing.set_padding(image)

        assert pytest.preprocessing.pad == (crop_size - size_x, crop_size - size_y)

        size_x = 100
        size_y = 166
        image = torch.Tensor(
            (np.random.rand(channels, size_y, size_x) * 255).astype("uint8")
        )

        # x greater than cropsize test
        pytest.preprocessing.set_padding(image)

        assert pytest.preprocessing.pad == (
            crop_size - size_x,
            crop_size - (size_y % crop_size),
        )

        size_x = 170
        size_y = 142
        image = torch.Tensor(
            (np.random.rand(channels, size_y, size_x) * 255).astype("uint8")
        )

        # both greater than cropsize test
        pytest.preprocessing.set_padding(image)

        assert pytest.preprocessing.pad == (
            crop_size - (size_x % crop_size),
            crop_size - size_y,
        )

        pytest.preprocessing.pad = None

        size_x = 170
        size_y = 166
        image = torch.Tensor(
            (np.random.rand(channels, size_y, size_x) * 255).astype("uint8")
        )

        pytest.preprocessing.set_padding(image)

        assert pytest.preprocessing.pad == (
            crop_size - (size_x % crop_size),
            crop_size - (size_y % crop_size),
        )

        size_x = 150
        size_y = 150
        image = torch.Tensor(
            (np.random.rand(channels, size_y, size_x) * 255).astype("uint8")
        )

        pytest.preprocessing.set_padding(image)

        assert pytest.preprocessing.pad == (0, 0)

    def test_scale(self):
        """Verify scale function"""
        size = 100
        channels = 3
        image = Image.fromarray(
            (np.random.rand(size, size, channels) * 255).astype("uint8")
        ).convert("RGB")
        target = np.random.randint(1, 10, (size, size)).astype("uint8")

        pytest.preprocessing.scale = 0.5
        result_image, result_target = pytest.preprocessing.scale_img(image, target)

        result_size = int(size * pytest.preprocessing.scale)
        assert result_image.shape == (channels, result_size, result_size)
        assert result_target.shape == (result_size, result_size)

        pytest.preprocessing.scale = 1
        result_image, result_target = pytest.preprocessing.scale_img(image, target)

        assert result_image.shape == (channels, size, size)
        assert result_target.shape == (size, size)

    def test_crop(self):
        """Verify crop_img."""
        size = 100
        channels = 4
        crop_size = 25
        image = np.random.rand(channels, size, size) * 255
        pytest.preprocessing.crop_size = crop_size
        pytest.preprocessing.crop_factor = 1

        windows = Preprocessing.crop_img(
            crop_size, 1, image
        )  # pylint: disable=protected-access

        assert windows.shape == (
            int(size / crop_size) ** 2,
            channels,
            crop_size,
            crop_size,
        )
