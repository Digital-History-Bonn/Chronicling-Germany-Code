"""Test class for reading xml scripts"""
import json

import numpy as np

from script.read_xml import read_transcribus
from script.draw_img import draw_img

DATA_PATH = './tests/data/'


class TestClassReadXML:
    """Class for testing xml converting"""
    def test_read(self):
        """tests read function of newspaper and HLNA Data"""
        result = read_transcribus(DATA_PATH + 'newspaper/test-annotation.xml')
        with open(DATA_PATH + 'newspaper/test-target.json', encoding="utf-8") as file:
            ground_truth = json.load(file)
            assert result == ground_truth

    def test_draw(self):
        """tests draw function of newspaper and HLNA Data"""
        with open(DATA_PATH + 'newspaper/test-target.json', encoding="utf-8") as file:
            result = draw_img(json.load(file))
        ground_truth = np.load(DATA_PATH + "newspaper/test-target.npy")
        assert (result == ground_truth).all()
