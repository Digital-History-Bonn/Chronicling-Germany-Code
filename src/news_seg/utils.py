"""Utility Module"""
import warnings
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from PIL.Image import BICUBIC  # pylint: disable=no-name-in-module
from numpy import ndarray
from torchmetrics.classification import MulticlassConfusionMatrix


def multi_class_csi(
        pred: torch.Tensor, target: torch.Tensor, metric: MulticlassConfusionMatrix
) -> torch.Tensor:
    """Calculate csi score using true positives, true negatives and false negatives from confusion matrix.
    Csi score is used as substitute for accuracy, calculated separately for each class.
    Returns numpy array with an entry for every class. If every prediction is a true negative,
    the score cant be calculated and the array will contain nan. These cases should be completely ignored.
    :param pred: prediction tensor
    :param target: target tensor
    :return:
    """
    pred = pred.flatten()
    target = target.flatten()

    matrix = metric(pred, target)
    true_positive = torch.diagonal(matrix)
    false_positive = torch.sum(matrix, axis=1) - true_positive
    false_negative = torch.sum(matrix, axis=0) - true_positive
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csi = torch.tensor(
            true_positive / (true_positive + false_negative + false_positive)
        )
    return csi


def get_file(file: str, scale: float = 0.25) -> torch.Tensor:
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
    return torch.unsqueeze(torch.tensor(img_t / 255, dtype=torch.float), dim=0)  # type: ignore


def replace_substrings(string: str, replacements: Dict[str, str]) -> str:
    """replaces substring (string) with replacements
    :param string: string with the substring(s) to replace
    :param replacements: dict with all substrings and there replacements
    :return: new string with replaced substrings
    """
    for substring, replacement in replacements.items():
        string = string.replace(substring, replacement)
    return string


def correct_shape(image: torch.Tensor) -> torch.Tensor:
    """
    If one of the dimension has an uneven number of pixels, the last row/ column is remove to achieve an
    even pixel number.
    :param image: input image
    :return: corrected image
    """
    if image.shape[2] % 2 != 0:
        image = image[:, :, :-1]
    if image.shape[1] % 2 != 0:
        image = image[:, :-1, :]
    return image


def create_bbox_ndarray(bbox_dict: Dict[int, List[List[float]]]) -> ndarray:
    """
    Takes Dict with label keys and bbox List and converts it to bbox ndarray.
    :param bbox_dict: Label keys and bbox Lists
    :return: 2d ndarray with n x 7 values. Containing id, label, 2 bbox corners and x-axis center.
    """
    index = 0
    result = []
    for label, bbox_list in bbox_dict.items():
        for bbox in bbox_list:
            result.append([index, label] + bbox)
            index += 1
    return np.array(result, dtype=int)


def calculate_x_axis_center(bbox: List[float]) -> float:
    """
    Calculate x-axis center from 2 1d points.
    :param bbox: bbox list containing x,y,max_x,max_y
    :return: center
    """
    return bbox[0] + abs(bbox[2] - bbox[0]) / 2
