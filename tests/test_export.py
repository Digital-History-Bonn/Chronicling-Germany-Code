"""Class for testing prediction and export scripts"""
import numpy as np

# pylint: disable=import-error
from bbox_test_data import bbox
from script.convert_xml import polygon_to_string
from script.transkribus_export import prediction_to_polygons, get_reading_order
from script.convert_xml import get_label_name
from src.news_seg import predict
from src.news_seg import utils


class TestClassExport:
    """Class for testing prediction and export scripts"""

    def test_process_prediction(self):
        """Function for testing prediction argmax and threshold handling.
        Each tripple of data represents probabilities for 3 possible classes.
        If the maximum is above the threshold, the result should contain that class label.
        Otherwise, it is always class 0."""
        data = np.transpose(
            np.array(
                [
                    [[0.1, 0.5, 0.4], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]],
                    [[0.0, 0.6, 0.4], [0.05, 0.05, 0.9], [0.01, 0.59, 0.4]],
                ]
            ),
            (2, 0, 1),
        )
        ground_truth = np.array([[0, 1, 1], [1, 2, 0]])

        result = predict.process_prediction(data, 0.6)
        assert np.all(result == ground_truth)

    def test_polygon_to_string(self):
        """Tests polygon list conversion to a coordinate string that transkribus can handle."""
        data = [19.0, 20.0, 1.0, 4.0, 5.5, 10.5, 20.0, 30.0]
        ground_truth = "19,20 1,4 5,10 20,30"

        assert polygon_to_string(data) == ground_truth

    def test_prediction_to_polygons(self):
        """Tests prediction conversion to a polygon list. Background pixels will not be converted to a polygon"""
        tolerance = [
            1.0,  # "UnknownRegion"
            1.0,  # "caption"
            1.0,  # "table"
            1.0,  # "article"
            1.0,  # "heading"
            1.0,  # "header"
            1.0,  # "separator_vertical"
            1.0,  # "separator_short"
            1.0]  # "separator_horizontal"

        data = np.array([[0, 0, 3, 3, 3], [0, 0, 3, 3, 1], [1, 1, 1, 1, 1]])
        ground_truth = (
        {1: [[4.0, 2.5, -0.5, 2.0, 4.0, 0.5, 4.0, 2.5]], 3: [[3.0, 1.5, 1.5, 1.0, 2.0, -0.5, 4.5, 0.0, 3.0, 1.5]]},
        {1: [[-0.5, 0.5, 4.0, 2.5]], 3: [[1.5, -0.5, 4.5, 1.5]]})
        assert prediction_to_polygons(data, tolerance) == ground_truth

    def test_get_label_names(self):
        """Tests prediction conversion to a polygon list. Background pixels will not be converted to a polygon"""
        assert get_label_name(1) == "UnknownRegion"
        assert get_label_name(4) == "article"
        assert get_label_name(9) == "separator_horizontal"

    def test_get_reading_order(self):
        """Tests reading order calculation based on bboxes. Elements contain id, label and bbox top left and
        bottom right corner"""
        bbox_data = np.array([[1, 4, 1, 1, 10, 10], [2, 9, 1, 100, 100, 105], [3, 3, 1, 11, 10, 21],
                              [4, 6, 15, 1, 25, 10], [5, 6, 15, 11, 25, 21], [6, 4, 1, 120, 10, 130],
                              [7, 4, 11, 120, 25, 130]])

        ground_truth = np.array([1, 3, 4, 5, 2, 6, 7])

        result = []
        get_reading_order(bbox_data, result)
        assert all(result == ground_truth)

        bbox_data = bbox
        ground_truth = np.array(
            [0, 11, 12, 1, 20, 2, 31, 3, 32, 22, 23, 25, 4, 5, 13, 6, 14, 24, 26, 28, 27, 21, 15, 29, 30, 7, 8, 9, 10,
             16, 17, 18, 19])

        result = []
        get_reading_order(bbox_data, result)
        assert all(result == ground_truth)

    def test_center(self):
        """
        Test x-axis center calculation from bbox list.
        """
        data = [10.0, 10.0, 20.0, 20.0]
        ground_thruth = 15.0

        assert utils.calculate_x_axis_center(data) == ground_thruth
