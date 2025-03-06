"""Test class for newsdataset"""
import json
from pathlib import Path

import numpy as np
# pylint: disable-next=no-name-in-module
# pylint: disable-next=import-error
import pytest
import torch
from bs4 import BeautifulSoup

from src.cgprocess.OCR.SSM.dataset import SSMDataset, extract_crop
from src.cgprocess.OCR.shared.tokenizer import OCRTokenizer
from src.cgprocess.OCR.shared.utils import create_unicode_alphabet
from src.cgprocess.shared.datasets import PageDataset
from src.cgprocess.layout_segmentation.datasets.train_dataset import TrainDataset
from src.cgprocess.layout_segmentation.processing.preprocessing import Preprocessing

DATA_PATH = "./tests/data/newsdataset/"
CONFIG_PATH = "./tests/ressources/test_config.yml/"


def test_extract_crop() -> None:
    """Tests extract crop method by loading ground truth from disc and comparing with the generated crops."""
    image = torch.ones(1, 1200, 1000, dtype=torch.uint8) * 5
    crop_height = 64
    result_list = []

    with open(f"{DATA_PATH}additional/single_test_line.xml", "r", encoding="utf-8") as file:
        data = file.read()
    soup = BeautifulSoup(data, 'xml')
    text_line = soup.find_all('TextLine')[0]

    extract_crop(result_list, image, text_line, crop_height)

    ground_truth = np.load(f"{DATA_PATH}additional/test_line_target.npy")

    assert ground_truth.shape == result_list[0].shape
    assert np.all(result_list[0] == (ground_truth * 5))

class TestOCRDataset:
    """Test class for OCRDataset"""
    @pytest.fixture(autouse=True, scope="class")
    def setup(self):
        """will initiate NewsDataset for every test"""
        args = {"data_path":Path(DATA_PATH)}
        pytest.image_height = 32
        pytest.ocr_dataset = SSMDataset(
            args,
            pytest.image_height,
            OCRTokenizer(create_unicode_alphabet(128)),
            num_processes=1,
        )

    def test_init(self) -> None:
        """verify file names list and data shapes"""
        with open(f"{DATA_PATH}output/file_names.json", encoding="utf-8") as file:
            ground_truth = json.load(file)
        file_quantity = 30

        assert pytest.ocr_dataset.file_stems == ground_truth
        assert len(pytest.ocr_dataset.file_stems) == file_quantity

        assert pytest.ocr_dataset[0][0] == torch.float
        assert pytest.ocr_dataset[0][0].shape[0] == pytest.image_height

        assert pytest.ocr_dataset[0][1] == torch.uint8
        assert pytest.ocr_dataset[0][2].dim == 1


class TestLayoutDataset:
    """Class for testing newsdataset"""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self):
        """will initiate NewsDataset for every test"""
        image_path = f"{DATA_PATH}input/"
        pytest.page_dataset = PageDataset(image_path)
        pytest.news_dataset = TrainDataset(
            Preprocessing(crop_size=256, crop_factor=1.5),
            image_path=f"{DATA_PATH}input/",
            target_path=f"{DATA_PATH}target_data/",
            sort=True,
            file_stems=pytest.page_dataset.file_stems,
        )

    def test_init(self) -> None:
        """verify file names list and length"""
        with open(f"{DATA_PATH}output/file_names.json", encoding="utf-8") as file:
            ground_truth = json.load(file)
        file_quantity = 30

        assert (
                pytest.news_dataset.file_stems == ground_truth
                and len(pytest.news_dataset.file_stems) == file_quantity
        )
        assert (pytest.news_dataset.data[0].dtype == torch.uint8
            and pytest.news_dataset.data[0].shape == (4, 256, 256)
        )

    def test_getitem(self) -> None:
        """Verify get_item. Particulary important is, that data ist in the right format.
        For example, RGB Values from 0 to 1 for images"""
        pytest.news_dataset.augmentations = False
        news_data = []
        news_targets = []
        for i, data in enumerate(pytest.news_dataset):
            if i > 3:
                break
            news_data.append(data[0])
            news_targets.append(data[1])
        news_data = torch.stack(news_data)
        news_targets = torch.stack(news_targets)
        # ground_truth_data = torch.load(f"{DATA_PATH}output/news_data.pt")
        # ground_truth_tragets = torch.load(f"{DATA_PATH}output/news_targets.pt")

        # assert torch.all(torch.eq(ground_truth_data, news_data))
        # assert torch.all(torch.eq(ground_truth_tragets, news_targets))
        assert news_data[0].dtype == torch.float
        assert news_targets[0].dtype == torch.long

    def test_split(self) -> None:
        """verify splitting operation"""

        page_dataset_1, page_dataset_2, page_dataset_3 = pytest.page_dataset.random_split(
            (0.5, 0.3, 0.2)
        )
        assert len(page_dataset_1) == 15 and len(page_dataset_2) == 9 and len(page_dataset_3) == 6

        dataset_1 = TrainDataset(
            Preprocessing(crop_size=256, crop_factor=1.5),
            image_path=f"{DATA_PATH}input/",
            target_path=f"{DATA_PATH}target_data/",
            sort=True,
            file_stems=page_dataset_1.file_stems,
            name="train"
        )

        dataset_2 = TrainDataset(
            Preprocessing(crop_size=256, crop_factor=1.5),
            image_path=f"{DATA_PATH}input/",
            target_path=f"{DATA_PATH}target_data/",
            sort=True,
            file_stems=page_dataset_1.file_stems,
            name="train"
        )

        dataset_3 = TrainDataset(
            Preprocessing(crop_size=256, crop_factor=1.5),
            image_path=f"{DATA_PATH}input/",
            target_path=f"{DATA_PATH}target_data/",
            sort=True,
            file_stems=page_dataset_1.file_stems,
            name="train"
        )

        assert dataset_1.data[0].shape == (4, 256, 256)
        assert dataset_2.data[0].shape == (4, 256, 256)
        assert dataset_3.data[0].shape == (4, 256, 256)
