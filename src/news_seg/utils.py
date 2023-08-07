"""Utility Module"""
import warnings
from typing import Dict

import numpy as np
import sklearn  # type: ignore
import torch
from numpy import ndarray
from PIL import Image  # type: ignore
from PIL.Image import BICUBIC  # pylint: disable=no-name-in-module


def multi_class_csi(
    pred: torch.Tensor, true: torch.Tensor, classes: int = 10
) -> ndarray:
    """Calculate csi score using true positives, true negatives and false negatives from confusion matrix.
    Csi score is used as substitute for accuracy, calculated separately for each class.
    Returns numpy array with an entry for every class. If every prediction is a true negative,
    the score cant be calculated and the array will contain nan. These cases should be completely ignored.
    :param pred: prediction tensor
    :param true: target tensor
    :param classes: number of possible classes
    :return:
    """
    pred = pred.flatten()
    true = true.flatten()
    matrix = sklearn.metrics.confusion_matrix(true, pred, labels=range(0, classes))
    true_positive = np.diagonal(matrix)
    false_positive = np.sum(matrix, axis=1) - true_positive
    false_negative = np.sum(matrix, axis=0) - true_positive
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csi = np.array(
            true_positive / (true_positive + false_negative + false_positive)
        )
    return csi


def get_file(file: str, scale=0.25) -> torch.Tensor:
    """
    loads a image as tensor
    :param file: path to file
    :param scale: scale
    :return: image as torch.Tensor
    """
    img = Image.open(file).convert("RGB")
    shape = int(img.size[0] * scale), int(img.size[1] * scale)
    img = img.resize(shape, resample=BICUBIC)

    w_pad, h_pad = (32 - (shape[0] % 32)), (32 - (shape[1] % 32))
    img_np = np.pad(
        np.asarray(img), ((0, h_pad), (0, w_pad), (0, 0)), "constant", constant_values=0
    )
    img_t = np.transpose(torch.tensor(img_np), (2, 0, 1))
    return torch.unsqueeze(torch.tensor(img_t / 255, dtype=torch.float), dim=0)


def replace_substrings(string: str, replacements: Dict[str, str]):
    """replaces substring (string) with replacements
    :param string: string with the substring(s) to replace
    :param replacements: dict with all substrings and there replacements
    :return: new string with replaced substrings
    """
    for substring, replacement in replacements.items():
        string = string.replace(substring, replacement)
    return string
