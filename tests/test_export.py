"""Class for testing prediction and export scripts"""
import numpy as np

import predict


class TestClassExport:
    """Class for testing prediction and export scripts"""

    def test_process_prediction(self):
        """class for testing prediction argmax and treshold handling"""
        data = np.transpose(np.array([[[0.1, 0.5, 0.4], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]],
                                      [[0.0, 0.6, 0.4], [0.05, 0.05, 0.9], [0.01, 0.59, 0.4]]]), (2, 0, 1))
        ground_truth = np.array([[0, 1, 1], [1, 2, 0]])

        result = predict.process_prediction(data, 0.6)
        assert np.all(result == ground_truth)
