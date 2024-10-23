"""Class for testing prediction and export scripts"""
import numpy as np
import torch

from src.cgprocess.layout_segmentation.predict import process_prediction_debug, process_prediction
from src.cgprocess.layout_segmentation.processing.polygon_handler import bbox_sufficient, uncertainty_to_polygons, \
    prediction_to_region_polygons
from src.cgprocess.layout_segmentation.processing.slicing_export import area_sufficient
from src.cgprocess.layout_segmentation.processing.transkribus_export import get_label_name, polygon_to_string
from src.cgprocess.layout_segmentation.utils import calculate_x_axis_center


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
        ground_truth = np.array([[0, 1, 1], [1, 2, 0]], dtype=np.uint8)

        result = process_prediction(torch.tensor(data[None, :, :, :]), 0.6)
        assert np.all(result == ground_truth)

    def test_process_debug_prediction(self):
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
        target = np.transpose(np.array([[1, 1, 1], [2, 2, 0]]), (0, 1))
        ground_truth = np.array([[1, 0, 0], [1, 0, 1]], dtype=np.uint8)

        result = process_prediction_debug(torch.tensor(data[None, :, :, :]),
                                          torch.tensor(target[None, None, :, :]),
                                          0.6)
        assert np.all(result == ground_truth)

    def test_polygon_to_string(self):
        """Tests polygon list conversion to a coordinate string that transkribus can handle."""
        data = [19.0, 20.0, 1.0, 4.0, 5.5, 10.5, 20.0, 30.0]
        ground_truth = "19,20 1,4 5,10 20,30"

        assert polygon_to_string(data, 1.0) == ground_truth

        ground_truth = "38,40 2,8 11,21 40,60"

        assert polygon_to_string(data, 2) == ground_truth

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
        assert prediction_to_region_polygons(data, tolerance, 1, True) == ground_truth

        data = np.array([[0, 0, 3, 3, 3], [0, 0, 3, 3, 2], [2, 2, 2, 2, 2]])
        ground_truth = (
            {2: [[4.0, 2.5, -0.5, 2.0, 4.0, 0.5, 4.0, 2.5]], 3: [[3.0, 1.5, 1.5, 1.0, 2.0, -0.5, 4.5, 0.0, 3.0, 1.5]]},
            {2: [[-0.5, 0.5, 4.0, 2.5]], 3: [[1.5, -0.5, 4.5, 1.5]]})
        assert prediction_to_region_polygons(data, tolerance, 1, True) == ground_truth

        ground_truth = ({2: [[4.0, 2.5, -0.5, 2.0, 4.0, 0.5, 4.0, 2.5]], 3: []}, {2: [[-0.5, 0.5, 4.0, 2.5]], 3: []})
        assert prediction_to_region_polygons(data, tolerance, 5, True) == ground_truth

    def test_debug_to_polygons(self):
        """Tests prediction conversion to a polygon list. Background pixels will not be converted to a polygon"""

        data = np.array([[0, 0, 1, 1, 1], [0, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=np.uint8)
        ground_truth = (
            {0: [], 1: [[2, 0, 2, 1, 1, 2, 0, 2, 4, 2, 4, 0]], 2: [[2, 1, 3, 0, 4, 1, 3, 2]], 3: [], 4: [], 5: [],
             6: [],
             7: [], 8: [], 9: []},
            {0: [], 1: [[4, 0, 0, 2]], 2: [[4, 0, 2, 2]], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []})
        assert uncertainty_to_polygons(data) == ground_truth

    def test_get_label_names(self):
        """Tests prediction conversion to a polygon list. Background pixels will not be converted to a polygon"""
        assert get_label_name(1) == "caption"
        assert get_label_name(2) == "table"
        assert get_label_name(3) == "paragraph"
        assert get_label_name(4) == "heading"
        assert get_label_name(5) == "header"
        assert get_label_name(6) == "separator_vertical"
        assert get_label_name(7) == "separator_horizontal"
        assert get_label_name(8) == "image"
        assert get_label_name(9) == "inverted_text"

    def test_center(self):
        """
        Test x-axis center calculation from bbox list.
        """
        data = [10.0, 10.0, 20.0, 20.0]
        ground_thruth = 15.0

        assert calculate_x_axis_center(data) == ground_thruth

    def test_bbox_sufficient(self):
        """Test bbox threshold"""
        data = [10.0, 10.0, 20.0, 20.0]
        assert bbox_sufficient(data, 19)
        assert not bbox_sufficient(data, 20)
        assert data == [10.0, 10.0, 20.0, 20.0]

    def test_area_sufficient(self):
        """Test bbox threshold"""
        data = [10.0, 10.0, 20.0, 20.0]
        assert area_sufficient(data, 99)
        assert not area_sufficient(data, 100)
        assert data == [10.0, 10.0, 20.0, 20.0]
