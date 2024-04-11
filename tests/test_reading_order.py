"""Class for testing prediction and export scripts"""
import numpy as np
import torch

from tests.bbox_test_data import bbox
from script.convert_xml import polygon_to_string
from script.transkribus_export import prediction_to_polygons, bbox_sufficient
from script.reading_order import get_splitting_regions, get_reading_order
from script.convert_xml import get_label_name
from src.news_seg import predict
from src.news_seg import utils


class TestClassExport:
    """Class for testing prediction and export scripts"""

    def test_get_reading_order(self):
        """Tests reading order calculation based on bboxes. Elements contain id, label and bbox top left and
        bottom right corner"""
        bbox_data = np.array([[1, 4, 1, 1, 10, 10], [2, 9, 1, 100, 100, 105], [3, 3, 1, 11, 10, 21],
                              [4, 6, 15, 1, 25, 10], [5, 6, 15, 11, 25, 21], [6, 4, 1, 120, 10, 130],
                              [7, 4, 11, 120, 25, 130], [8, 9, 1, 200, 100, 205], [9, 9, 1, 210, 100, 215]])

        ground_truth = np.array([1, 3, 4, 5, 2, 6, 7, 8, 9])

        result = []
        get_reading_order(bbox_data, result, 0)
        assert all(result == ground_truth)

        bbox_data = bbox
        ground_truth = np.array(
            [0, 11, 12, 1, 20, 2, 31, 3, 32, 22, 23, 25, 4, 5, 13, 6, 14, 24, 26, 28, 27, 21, 15, 29, 30, 7, 8, 9, 10,
             16, 17, 18, 19])

        result = []
        get_reading_order(bbox_data, result, 0)
        assert all(result == ground_truth)

    def test_get_splitting_regions(self):
        """Test spliting regions deection with threshold"""
        bbox_data = np.array([[1, 4, 1, 1, 10, 10], [2, 9, 1, 100, 100, 105], [3, 3, 1, 11, 10, 21],
                              [4, 6, 15, 1, 25, 10], [5, 6, 15, 11, 25, 21], [6, 4, 1, 120, 10, 130],
                              [7, 4, 11, 120, 25, 130], [8, 9, 1, 200, 100, 205], [9, 9, 1, 210, 101, 215]])
        ground_truth = np.array([1, 7, 8])  # actual indices are 2,8,9 but list begins with 0

        result = get_splitting_regions(bbox_data, 0)
        assert all(result == ground_truth)

        ground_truth = np.array([8])
        result = get_splitting_regions(bbox_data, 99)
        assert all(result == ground_truth)
