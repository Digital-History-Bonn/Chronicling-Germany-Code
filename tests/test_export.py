"""Class for testing prediction and export scripts"""
import numpy as np
import torch

from data.bbox_test_data import bbox
from script.convert_xml import polygon_to_string
from script.transkribus_export import prediction_to_polygons, bbox_sufficient
from script.reading_order import get_splitting_regions, get_reading_order
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
        ground_truth = np.array([[0, 1, 1], [1, 2, 0]], dtype=np.uint8)

        result = predict.process_prediction(torch.tensor(data[None, :, :, :]), 0.6)
        assert np.all(result == ground_truth)

    def test_polygon_to_string(self):
        """Tests polygon list conversion to a coordinate string that transkribus can handle."""
        data = [19.0, 20.0, 1.0, 4.0, 5.5, 10.5, 20.0, 30.0]
        ground_truth = "19,20 1,4 5,10 20,30"

        assert polygon_to_string(data, 1.0) == ground_truth

        ground_truth = "38,40 2,8 11,21 40,60"

        assert polygon_to_string(data, 0.5) == ground_truth

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
            {3: [[3.0, 1.5, 1.5, 1.0, 2.0, -0.5, 4.5, 0.0, 3.0, 1.5]]},
            {3: [[1.5, -0.5, 4.5, 1.5]]})
        assert prediction_to_polygons(data, tolerance, 1, True) == ground_truth

        data = np.array([[0, 0, 3, 3, 3], [0, 0, 3, 3, 2], [2, 2, 2, 2, 2]])
        ground_truth = (
            {2: [[4.0, 2.5, -0.5, 2.0, 4.0, 0.5, 4.0, 2.5]], 3: [[3.0, 1.5, 1.5, 1.0, 2.0, -0.5, 4.5, 0.0, 3.0, 1.5]]},
            {2: [[-0.5, 0.5, 4.0, 2.5]], 3: [[1.5, -0.5, 4.5, 1.5]]})
        assert prediction_to_polygons(data, tolerance, 1, True) == ground_truth

        ground_truth = ({2: [[4.0, 2.5, -0.5, 2.0, 4.0, 0.5, 4.0, 2.5]], 3: []}, {2: [[-0.5, 0.5, 4.0, 2.5]], 3: []})
        assert prediction_to_polygons(data, tolerance, 5, True) == ground_truth

    def test_get_label_names(self):
        """Tests prediction conversion to a polygon list. Background pixels will not be converted to a polygon"""
        assert get_label_name(1) == "UnknownRegion"
        assert get_label_name(4) == "article"
        assert get_label_name(9) == "separator_horizontal"

    def test_center(self):
        """
        Test x-axis center calculation from bbox list.
        """
        data = [10.0, 10.0, 20.0, 20.0]
        ground_thruth = 15.0

        assert utils.calculate_x_axis_center(data) == ground_thruth

    def test_bbox_sufficient(self):
        """Test bbox threshold"""
        data = [10.0, 10.0, 20.0, 20.0]
        assert bbox_sufficient(data, 19)
        assert not bbox_sufficient(data, 20)
        assert data == [10.0, 10.0, 20.0, 20.0]

    def test_area_sufficient(self):
        """Test bbox threshold"""
        data = [10.0, 10.0, 20.0, 20.0]
        assert predict.area_sufficient(data, 99)
        assert not predict.area_sufficient(data, 100)
        assert data == [10.0, 10.0, 20.0, 20.0]
