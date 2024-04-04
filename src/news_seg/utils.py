"""Utility Module"""
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from PIL.Image import BICUBIC  # pylint: disable=no-name-in-module
from matplotlib import pyplot as plt
from numpy import ndarray
from skimage.color import label2rgb # pylint: disable=no-name-in-module
from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision import transforms

from src.news_seg.class_config import LABEL_NAMES
from src.news_seg.class_config import cmap_12 as cmap

def draw_prediction(img: ndarray, path: str) -> None:
    """
    Draw prediction with legend. And save it.
    :param img: prediction ndarray
    :param path: path for the prediction to be saved.
    """

    # unique, counts = np.unique(img, return_counts=True)
    # print(dict(zip(unique, counts)))
    values = LABEL_NAMES
    for i in range(len(values)):
        img[-1][-(i + 1)] = i + 1
    plt.imshow(label2rgb(img, bg_label=0, colors=cmap))
    plt.axis("off")
    # create a patch (proxy artist) for every color
    # patches = [mpatches.Patch(color=cmap[i], label=f"{values[i]}") for i in range(9)]
    # put those patched as legend-handles into the legend
    # plt.legend(handles=patches, bbox_to_anchor=(1.3, -0.10), loc="lower right")
    plt.autoscale(tight=True)
    plt.savefig(path, bbox_inches=0, pad_inches=0, dpi=500)
    # plt.show()

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

    matrix: torch.Tensor = metric(pred, target)
    true_positive = torch.diagonal(matrix)
    false_positive = torch.sum(matrix, dim=1) - true_positive
    false_negative = torch.sum(matrix, dim=0) - true_positive
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csi = torch.tensor(
            true_positive / (true_positive + false_negative + false_positive)
        )
    return csi

def multi_precison_recall(
        pred: torch.Tensor, target: torch.Tensor, out_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate precision and recall using true positives, true negatives and false negatives from confusion matrix.
    Returns numpy array with an entry for every class. If every prediction is a true negative,
    the score cant be calculated and the array will contain nan. These cases should be completely ignored.
    :param pred: prediction tensor
    :param target: target tensor
    :return:
    """

    pred = torch.argmax(pred, dim=1).type(torch.uint8)

    metric: MulticlassConfusionMatrix = MulticlassConfusionMatrix(num_classes=out_channels).to(pred.get_device())

    pred = pred.flatten()
    target = target.flatten()

    # pylint: disable=not-callable
    matrix: torch.Tensor = metric(pred, target)
    true_positive = torch.diagonal(matrix)
    false_positive = torch.sum(matrix, dim=1) - true_positive
    false_negative = torch.sum(matrix, dim=0) - true_positive
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = torch.tensor(
            true_positive / (true_positive + false_positive)
        )
        recall = torch.tensor(
            true_positive / (true_positive + false_negative)
        )
    return precision, recall


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


def split_batches(tensor: torch.Tensor, permutation: Tuple[int, ...], num_scores_splits: int) -> torch.Tensor:
    """
    Splits tensor into self.num_scores_splits chunks. This is necessary to not overload the multiprocessing Queue.
    :param permutation: permutation for this tensor. On a tensor with feature dimensions,
    the feature dimension should be transferred to the end. Everything else has to stay in the same order.
    :param tensor: [B,C,H,W]
    :return: ndarray version of result
    """
    tensor = torch.permute(tensor, permutation).flatten(0, 2)
    return torch.stack(torch.split(tensor, tensor.shape[0] // num_scores_splits))  # type: ignore


def calculate_padding(pad: Tuple[int, int], shape: Tuple[int, ...], scale: float) -> Tuple[int, int]:
    """
    Calculate padding values to be added to the right and bottom of the image. It will make shure, that the
    padded image is divisible by crop size.
    :param image: tensor image
    :return: padding tuple for right and bottom
    """
    # pad = ((crop_size - (image.shape[1] % crop_size)) % crop_size,
    #        (crop_size - (image.shape[2] % crop_size)) % crop_size)
    pad = (int(pad[0] * scale), int(pad[1] * scale))

    assert (
            pad[1] >= shape[-2]
            and pad[0] >= shape[-1]
    ), (
        f"Final size has to be greater than actual image size. "
        f"Padding to {pad[0]} x {pad[1]} "
        f"but image has shape of {shape[-1]} x {shape[-2]}"
    )

    pad = (pad[1] - shape[-2], pad[0] - shape[-1])
    return pad


def pad_image(pad: Tuple[int, int], image: torch.Tensor) -> torch.Tensor:
    """
    Pad image to given size.
    :param pad: values to be added on the right and bottom.
    :param image: image tensor
    :return: padded image
    """
    # debug shape
    # print(image.shape)
    transform = transforms.Pad(
        (
            0,
            0,
            (pad[1]),
            (pad[0]),
        )
    )
    image = transform(image)
    # debug shape
    # print(image.shape)
    return image
