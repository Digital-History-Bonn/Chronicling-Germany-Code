"""Test class for newsdataset"""
import json

import pytest
import torch

from news_dataset import NewsDataset

DATA_PATH = './tests/data/newsdataset/'


class TestClassNewsdataset:
    """Class for testing newsdataset"""

    @pytest.fixture(autouse=True)
    def setup(self):
        "will initiate NewsDataset for every test"
        pytest.news_dataset = NewsDataset(path=f"{DATA_PATH}data/")
        pytest.news_dataset.file_names.sort()

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
        news_targets = []
        for data in pytest.news_dataset:
            news_data.append(data[0])
            news_targets.append(data[1])
        news_data = torch.cat(news_data)
        news_targets = torch.cat(news_targets)
        ground_truth_data = torch.load(f"{DATA_PATH}output/news_data.pt")
        ground_truth_tragets = torch.load(f"{DATA_PATH}output/news_targets.pt")

        assert torch.all(torch.eq(ground_truth_data, news_data))
        assert torch.all(torch.eq(ground_truth_tragets, news_targets))

    def test_split(self):
        """verify splitting operation"""
        dataset_1, dataset_2, dataset_3 = pytest.news_dataset.random_split((0.5, 0.3, 0.2))
        assert len(dataset_1) == 5 and len(dataset_2) == 3 and len(dataset_3) == 2
        try:
            dataset_1.augmentations = False
            dataset_2.augmentations = False
            dataset_2.augmentations = False
        except AttributeError as exc:
            assert False, f"random split does not result in Newsdatasets. Those are " \
                          f"expected to have an augmentations attribute {exc}"
