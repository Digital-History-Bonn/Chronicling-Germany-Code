"""Module for predicting newspaper images with trained models. """

import argparse
import os
from typing import Dict, List, Tuple, Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from PIL.Image import BICUBIC # pylint: disable=no-name-in-module
from numpy import ndarray
from skimage import draw
from skimage.color import label2rgb  # pylint: disable=no-name-in-module
from torchvision import transforms
from tqdm import tqdm

from script.convert_xml import create_xml
from script.draw_img import LABEL_NAMES
from script.transkribus_export import prediction_to_polygons, get_reading_order
from src.news_seg import train
from src.news_seg.utils import correct_shape, create_bbox_ndarray

# import train

DATA_PATH = "../../data/newspaper/input/"
RESULT_PATH = "../../data/output/"

FINAL_SIZE = (3200, 3200)

# Tolerance pixel for polygon simplification. All points in the simplified object will be within
# the tolerance distance of the original geometry.
TOLERANCE = [
    10.0,  # "UnknownRegion"
    10.0,  # "caption"
    10.0,  # "table"
    10.0,  # "article"
    5.0,  # "heading"
    10.0,  # "header"
    2.0,  # "separator_vertical"
    2.0,  # "separator_short"
    5.0]  # "separator_horizontal"

cmap = [
    (1.0, 0.0, 0.16),
    (1.0, 0.43843843843843844, 0.0),
    (0, 0.222, 0.222),
    (0.36036036036036045, 0.5, 0.5),
    (0.0, 1.0, 0.2389486260454002),
    (0.8363201911589008, 1.0, 0.0),
    (0.0, 0.5615942028985507, 1.0),
    (0.0422705314009658, 0.0, 1.0),
    (0.6461352657004831, 0.0, 1.0),
    (1.0, 0.0, 0.75),
]


def draw_prediction(img: ndarray, path: str) -> None:
    """
    Draw prediction with legend. And save it.
    :param img: prediction ndarray
    :param path: path for the prediction to be saved.
    """

    unique, counts = np.unique(img, return_counts=True)
    print(dict(zip(unique, counts)))
    values = LABEL_NAMES
    for i in range(len(values)):
        img[-1][-(i + 1)] = i + 1
    plt.imshow(label2rgb(img, bg_label=0, colors=cmap))
    plt.axis("off")
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=cmap[i], label=f"{values[i]}") for i in range(9)]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.3, -0.10), loc="lower right")
    plt.autoscale(tight=True)
    plt.savefig(path, bbox_inches=0, pad_inches=0, dpi=500)
    # plt.show()


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument(
        "--data-path",
        "-p",
        type=str,
        default=DATA_PATH,
        help="path for folder with images to be segmented. Images need to be png or jpg. Otherwise they"
        " will be skipped",
    )
    parser.add_argument(
        "--result-path",
        "-r",
        type=str,
        default=RESULT_PATH,
        help="path for folder where prediction images are to be saved",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default="model.pt",
        help="path to model .pt file",
    )
    parser.add_argument(
        "--transkribus-export",
        "-e",
        dest="export",
        action="store_true",
        help="If True, annotation data ist added to xml files inside the page folder. The page folder "
        "needs to be inside the image folder.",
    )
    parser.add_argument(
        "--cuda-device", "-c", type=str, default="cuda:0", help="Cuda device string"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Confidence threshold for assigning a label to a pixel.",
    )
    parser.add_argument(
        "--model-architecture",
        "-a",
        type=str,
        default="dh_segment",
        help="which model to load options are 'dh_segment, trans_unet, dh_segment_small",
    )
    parser.add_argument(
        "--final_size",
        "-s",
        type=int,
        nargs="+",
        default=FINAL_SIZE,
        help="Size to which the image will be padded to. Has to be a tuple (W, H). "
        "Has to be grater or equal to actual image",
    )
    parser.add_argument(
        "--torch-seed", "-ts", type=float, default=314.0, help="Torch seed"
    )
    parser.add_argument(
        "--scale",
        type=float,
        dest="scale",
        default=1,
        help="Downscaling factor of the images. Polygon data will be upscaled accordingly",
    )
    return parser.parse_args()


def load_image(file: str, args: argparse.Namespace) -> torch.Tensor:
    """
    Loads image and applies necessary transformation for prdiction.
    :param file: path to image
    :return: Tensor of dimensions (BxCxHxW). In this case, the number of batches will always be 1.
    """
    image = Image.open(args.data_path + file).convert("RGB")
    shape = int(image.size[0] * args.scale), int(image.size[1] * args.scale)
    image = image.resize(shape, resample=BICUBIC)
    transform = transforms.PILToTensor()
    data: torch.Tensor = transform(image).float() / 255
    data = torch.unsqueeze(data, dim=0)
    return data


def predict(args: argparse.Namespace) -> None:
    """
    Loads all images from the data folder and predicts segmentation.
    """
    device = args.cuda_device if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    file_names = os.listdir(args.data_path)
    model = train.init_model(args.model_path, device, args.model_architecture)
    model.to(device)
    model.eval()
    for file in tqdm(
        file_names, desc="predicting images", total=len(file_names), unit="files"
    ):
        if os.path.splitext(file)[1] != ".png" and os.path.splitext(file)[1] != ".jpg":
            continue
        image = load_image(file, args)
        args.final_size[1] = int(args.final_size[1] * args.scale)
        args.final_size[0] = int(args.final_size[0] * args.scale)
        assert (
            args.final_size[1] >= image.shape[2]
            and args.final_size[0] >= image.shape[3]
        ), (
            f"Final size has to be greater than actual image size. "
            f"Padding to {args.final_size} x {args.final_size} "
            f"but image has shape of {image.shape[3]} x {image.shape[2]}"
        )

        image = pad_image(args, image)

        execute_prediction(args, device, file, image, model)


def execute_prediction(args: argparse.Namespace, device: str, file: str, image: torch.Tensor, model: Any) -> None:
    """
    Run model to create prediction and call export methods.
    :param args:
    :param device:
    :param file:
    :param image:
    :param model:
    """
    pred = torch.nn.functional.softmax(
        torch.squeeze(model(image.to(device)).detach().cpu()), dim=0
    ).numpy()
    pred = process_prediction(pred, args.threshold)
    draw_prediction(pred, args.result_path + os.path.splitext(file)[0] + ".png")
    export_polygons(file, pred, args)


def pad_image(args: argparse.Namespace, image: torch.Tensor) -> torch.Tensor:
    """
    Pad image to given size.
    :param args: arguments
    :param image: image tensor
    :return: padded image
    """
    image = correct_shape(torch.squeeze(image))[None, :]
    # debug shape
    # print(image.shape)
    transform = transforms.Pad(
        (
            (args.final_size[0] - image.shape[3]) // 2,
            (args.final_size[1] - image.shape[2]) // 2,
        )
    )
    image = transform(image)
    # debug shape
    # print(image.shape)
    return image


def export_polygons(file: str, pred: ndarray, args: argparse.Namespace) -> None:
    """
    Simplify prediction to polygons and export them to an image as well as transcribus xml
    :param file: path
    :param pred: prediction 2d ndarray
    """
    if args.export:
        polygon_pred, reading_order_dict, segmentations = get_polygon_prediction(pred)

        draw_prediction(
            polygon_pred,
            args.result_path + f"{os.path.splitext(file)[0]}_polygons" + ".png",
        )

        with open(
            f"{args.data_path}page/{os.path.splitext(file)[0]}.xml",
            "r",
            encoding="utf-8",
        ) as xml_file:
            xml_data = create_xml(xml_file.read(), segmentations, reading_order_dict, args.scale)
        with open(
            f"{args.data_path}page/{os.path.splitext(file)[0]}.xml",
            "w",
            encoding="utf-8",
        ) as xml_file:
            xml_file.write(xml_data.prettify())


def get_polygon_prediction(pred: ndarray) -> Tuple[ndarray, Dict[int, int], Dict[int, List[List[float]]]]:
    """
    Calls polyong conversion twice. Original segmentation is first converted to polygons, then those polygons are
    drawen into an ndarray image. This smothed prediction is again converted to polygons which are used to
    determine reading order.
    :param pred: Original prediction ndarray image
    :return: smothed prediction ndarray image, reading order and segmentation dictionary
    """
    segmentations, bbox_list = prediction_to_polygons(pred, TOLERANCE)
    polygon_pred = draw_polygons(segmentations, pred.shape)
    segmentations, bbox_list = prediction_to_polygons(polygon_pred, TOLERANCE)

    bbox_ndarray = create_bbox_ndarray(bbox_list)
    reading_order: List[int] = []
    get_reading_order(bbox_ndarray, reading_order)
    reading_order_dict = {k: v for v, k in enumerate(reading_order)}

    return polygon_pred, reading_order_dict, segmentations


def draw_polygons(
    segmentations: Dict[int, List[List[float]]], shape: Tuple[int, ...]
) -> ndarray:
    """
    Takes segmentation dictionary and draws polygons with assigned labels into a new image.
    :param shape: shape of original image
    :param segmentations: dictionary assigning labels to polygon lists
    :return: result image as ndarray
    """

    polygon_pred = np.zeros(shape, dtype="uint8")
    for label, segmentation in segmentations.items():
        for polygon in segmentation:
            polygon_ndarray = np.reshape(polygon, (-1, 2)).T
            x_coords, y_coords = draw.polygon(polygon_ndarray[1], polygon_ndarray[0])
            polygon_pred[x_coords, y_coords] = label
    return polygon_pred


def process_prediction(pred: ndarray, threshold: float) -> ndarray:
    """
    Apply argmax to prediction and assign label 0 to all pixel that have a confidence below the threshold.
    :param threshold: confidence threshold for prediction
    :param pred: prediction
    :return:
    """
    argmax: ndarray = np.argmax(pred, axis=0)
    argmax[np.max(pred, axis=0) < threshold] = 0
    return argmax


if __name__ == "__main__":
    parameter_args = get_args()
    torch.manual_seed(parameter_args.torch_seed)
    predict(parameter_args)
