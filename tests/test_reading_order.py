"""Class for testing prediction and export scripts"""
import json

import numpy as np

from src.news_seg.processing.reading_order import get_global_splitting_regions, get_local_splitting_regions, \
    PageProperties

DATA_PATH = "./tests/data/"


class TestClassExport:
    """Class for testing prediction and export scripts"""

    def test_get_reading_order(self):
        """Tests reading order calculation based on bboxes. Elements contain id, label and bbox top left and
        bottom right corner"""
        with open(DATA_PATH + "export/reading_order.json", encoding="utf-8") as file:
            region_properties = np.array(json.load(file))

        ground_truth = {0: 20, 1: 18, 2: 3, 3: 7, 4: 9, 5: 11, 6: 2, 7: 5, 8: 1, 9: 19, 10: 16, 11: 6, 12: 8, 13: 10,
                        14: 4, 15: 12, 16: 13, 17: 15, 18: 17, 19: 0, 20: 14}

        page = PageProperties({}, region_properties)
        result = page.get_reading_order()
        assert result == ground_truth

    def test_splitting_regions(self):
        """Test spliting regions detection with threshold"""
        properties = np.array([[1, 4, 1, 1, 10, 10], [2, 7, 1, 100, 50, 20], [3, 3, 1, 11, 10, 21],
                               [4, 6, 15, 1, 25, 10], [5, 5, 10, 40, 25, 21], [6, 4, 1, 120, 10, 130],
                               [7, 4, 11, 120, 25, 130], [8, 9, 30, 100, 65, 205], [9, 9, 1, 110, 55, 215]])
        # does not contain width and heigth, as this is not required for this function

        global_ground_truth = np.array([1, 8])  # actual indices are 2,9 but list begins with 0
        local_ground_truth = np.array([7])
        x_median = 25
        columns_per_page = 4

        result = np.where(get_global_splitting_regions(properties, x_median, columns_per_page))[0].tolist()
        assert len(result) == len(global_ground_truth)
        assert all(result == global_ground_truth)

        result = np.where(get_local_splitting_regions(properties, x_median, columns_per_page))[0].tolist()
        assert len(result) == len(local_ground_truth)
        assert all(result == local_ground_truth)

        columns_per_page = 3
        global_ground_truth = np.array([1, 7, 8])
        result = np.where(get_global_splitting_regions(properties, x_median, columns_per_page))[0].tolist()
        assert len(result) == len(global_ground_truth)
        assert all(result == global_ground_truth)

        local_ground_truth = np.array([])
        result = np.where(get_local_splitting_regions(properties, x_median, columns_per_page))[0].tolist()
        assert len(result) == len(local_ground_truth)
        assert all(result == local_ground_truth)

        columns_per_page = 5
        global_ground_truth = np.array([])
        result = np.where(get_global_splitting_regions(properties, x_median, columns_per_page))[0].tolist()
        assert len(result) == len(global_ground_truth)
        assert all(result == global_ground_truth)

        local_ground_truth = np.array([1, 7, 8])
        result = np.where(get_local_splitting_regions(properties, x_median, columns_per_page))[0].tolist()
        assert len(result) == len(local_ground_truth)
        assert all(result == local_ground_truth)

        x_median = 20
        global_ground_truth = np.array([1, 8])
        result = np.where(get_global_splitting_regions(properties, x_median, columns_per_page))[0].tolist()
        assert len(result) == len(global_ground_truth)
        assert all(result == global_ground_truth)

        local_ground_truth = np.array([4, 7])
        result = np.where(get_local_splitting_regions(properties, x_median, columns_per_page))[0].tolist()
        assert len(result) == len(local_ground_truth)
        assert all(result == local_ground_truth)
