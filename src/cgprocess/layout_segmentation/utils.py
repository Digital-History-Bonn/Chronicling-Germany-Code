"""Utility Module"""

import argparse
from multiprocessing import Queue
from typing import Dict, List, Optional, Tuple

import torch

# from PIL.Image import BICUBIC  # pylint: disable=no-name-in-module # type:ignore
from matplotlib import pyplot as plt
from numpy import ndarray
from skimage.color import label2rgb  # pylint: disable=no-name-in-module
from torchvision import transforms

from src.cgprocess.layout_segmentation.class_config import (
    LABEL_NAMES,
    REDUCE_CLASSES,
    cmap,
)


def adjust_path(path: Optional[str]) -> Optional[str]:
    """
    Make sure, there is a slash at the end of a (folder) spath string.
    """
    return path if not path or path[-1] == "/" else path + "/"


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
    # patches = [mpatches.Patch(color=cmap[i], label=f"{values[i]}") for i in range(len(LABEL_NAMES))]
    # put those patched as legend-handles into the legend
    # plt.legend(handles=patches, bbox_to_anchor=(1.3, -0.10), loc="lower right")
    plt.autoscale(tight=True)
    plt.savefig(path, bbox_inches=0, pad_inches=0, dpi=500)
    plt.clf()
    del img
    # plt.show()


def replace_substrings(string: str, replacements: Dict[str, str]) -> str:
    """replaces substring (string) with replacements
    :param string: string with the substring(s) to replace
    :param replacements: dict with all substrings and there replacements
    :return: new string with replaced substrings
    """
    for substring, replacement in replacements.items():
        string = string.replace(substring, replacement)
    return string


def calculate_x_axis_center(bbox: List[float]) -> float:
    """
    Calculate x-axis center from 2 1d points.
    :param bbox: bbox list containing x,y,max_x,max_y
    :return: center
    """
    return bbox[0] + abs(bbox[2] - bbox[0]) / 2


def split_batches(
    tensor: torch.Tensor, permutation: Tuple[int, ...], num_scores_splits: int
) -> torch.Tensor:
    """
    Splits tensor into self.num_scores_splits chunks. This is necessary to not overload the multiprocessing Queue.
    :param permutation: permutation for this tensor. On a tensor with feature dimensions,
    the feature dimension should be transferred to the end. Everything else has to stay in the same order.
    :param tensor: [B,C,H,W]
    :return: ndarray version of result
    """
    tensor = torch.permute(tensor, permutation).flatten(0, 2)
    return torch.stack(torch.split(tensor, tensor.shape[0] // num_scores_splits))  # type: ignore


def calculate_padding(
    pad: Tuple[int, int], shape: Tuple[int, ...], scale: float
) -> Tuple[int, int]:
    """
    Calculate padding values to be added to the right and bottom of the image.
    :param image: tensor image
    :return: padding tuple for right and bottom
    """
    # pad = ((crop_size - (image.shape[1] % crop_size)) % crop_size,
    #        (crop_size - (image.shape[2] % crop_size)) % crop_size)
    pad = (int(pad[0] * scale), int(pad[1] * scale))

    assert pad[1] >= shape[-2] and pad[0] >= shape[-1], (
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


def replace_labels(target: torch.Tensor) -> torch.Tensor:
    """
    Replace labels to reduce classes
    """
    for replace_label, label_list in REDUCE_CLASSES.items():
        for label in label_list:
            target[target == label] = replace_label
    return target


def collapse_prediction(pred: torch.Tensor) -> torch.Tensor:
    """
    Collapses classes in the prediction tensor after softmax activation.
    This is used to make models with different classes compatible. This does not change the total number of classes.
    """
    for replace_label, label_list in REDUCE_CLASSES.items():
        for label in label_list:
            pred[:, replace_label, :, :] += pred[:, label, :, :]
            pred[:, label, :, :] = 0
    return pred


def create_model_list(
    args: argparse.Namespace, num_gpus: int, num_processes: int
) -> List[tuple]:
    """
    Create list of tuples that contain all information for model initialization. Tuple is used as arguments for
    train_helper.model_init()
    """
    models = (
        [
            (
                args.model_path,
                f"cuda:{i % num_gpus}",
                args.model_architecture,
                False,
                args.skip_cbam,
            )
            for i in range(num_gpus * num_processes)
        ]
        if torch.cuda.is_available()
        else [
            (
                args.model_path,
                "cpu",
                args.model_architecture,
                False,
                args.skip_cbam,
            )
        ]
    )
    return models


def create_path_queue(
    file_names: List[str], args: argparse.Namespace, dataset: object
) -> Queue:
    """
    Creates and fills path queue with image paths to be predicted. Elements are required to have the image path at
    index 0 and the bool variable for terminating processes at index -1.
    """
    path_queue: Queue = Queue()
    for file_name in file_names:
        path_queue.put((file_name, args, dataset, False))
    return path_queue
