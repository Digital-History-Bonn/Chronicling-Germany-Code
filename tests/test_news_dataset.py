"""Test class for newsdataset"""
import json

import pytest

from news_dataset import NewsDataset

DATA_PATH = './data/newsdataset/'


class TestClassNewsDataset:
    """Class for testing newsdataset"""

    @pytest.fixture(autouse=True)
    def setup(self):
        "will initiate NewsDataset for every test"
        pytest.news_dataset = NewsDataset(path=f"{DATA_PATH}/data/")

    def test_init(self):
        """verify file names list and length"""
        with open(f"{DATA_PATH}output/file_names.json", encoding="utf-8") as file:
            ground_truth = json.load(file)
            assert pytest.news_dataset.file_names == ground_truth and len(pytest.news_dataset) == 10

    def test_getitem(self):
        """Verify get_item. Particulary important is, that data ist in the right format.
        For example, RGB Values from 0 to 1 for images """
        pytest.news_dataset.augmentations = False
        news_data = []
        for data in pytest.news_dataset:
            news_data.append(data)
        with open(f"{DATA_PATH}output/news_data.json", mode="w", encoding="utf-8") as file:
            json.dump(news_data, file)
        # with open(f"{DATA_PATH}output/file_names.json", encoding="utf-8") as file:
        #     ground_truth = json.load(file)
        #     assert self.news_dataset == ground_truth and self.news_dataset.len == 10
