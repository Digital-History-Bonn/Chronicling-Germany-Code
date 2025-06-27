"""Test class for reading xml scripts"""

import json

import numpy as np
from bs4 import BeautifulSoup

from cgprocess.layout_segmentation.utils import draw_prediction
from src.cgprocess.layout_segmentation.processing.draw_img_from_polygons import draw_img
from src.cgprocess.layout_segmentation.processing.read_xml import read_transkribus
from src.cgprocess.shared.utils import xml_polygon_to_polygon_list

DATA_PATH = "tests/data/"


class TestClassReadXML:
    """Class for testing xml converting"""

    def test_read(self):
        """tests read function of newspaper and HLNA Data"""
        result = read_transkribus(DATA_PATH + "newspaper/test-annotation.xml", True)
        with open(DATA_PATH + "newspaper/test-target.json", encoding="utf-8") as file:
            ground_truth = json.load(file)
            assert result == ground_truth

    def test_draw(self):
        """tests draw function of newspaper and HLNA Data"""
        with open(DATA_PATH + "newspaper/test-target.json", encoding="utf-8") as file:
            result = draw_img(json.load(file))
        # draw_prediction(result, f"{DATA_PATH}newspaper/test-img.png")
        ground_truth = np.load(DATA_PATH + "newspaper/test-target.npy")
        assert (result == ground_truth).all()

    def test_polygon_to_string(self):
        """Tests polygon list conversion to a coordinate string that transkribus can handle."""
        with open(DATA_PATH + "read/tag.xml", "r", encoding="utf-8") as file:
            data = file.read()

        bs_data = BeautifulSoup(data, "xml")
        tag = bs_data.find_all("TextRegion")[0]

        ground_truth = [[3498, 5612], [4581, 5626], [4594, 6971], [3506, 6955]]

        assert xml_polygon_to_polygon_list(tag.Coords["points"]) == ground_truth
