"""Module for predicting newspaper images with trained models. """

import argparse
import os
from typing import Dict, List, Tuple, Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from PIL.Image import BICUBIC  # pylint: disable=no-name-in-module
from numpy import ndarray
from skimage import draw
from skimage.color import label2rgb  # pylint: disable=no-name-in-module
from torchvision import transforms
from tqdm import tqdm
# from torch.utils.data import DataLoader

from script.convert_xml import create_xml
from script.draw_img import LABEL_NAMES
from script.transkribus_export import prediction_to_polygons, get_reading_order
from src.news_seg import train
from src.news_seg.utils import create_bbox_ndarray

# from src.news_seg.preprocessing import Preprocessing
# from src.news_seg.train import OUT_CHANNELS

# import train

DATA_PATH = "../../data/newspaper/input/"
RESULT_PATH = "../../data/output/"

CROP_SIZE = 1024
FINAL_SIZE = (1024, 1024)

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
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default=DATA_PATH,
        help="path for folder with images to be segmented. Images need to be png or jpg. Otherwise they"
             " will be skipped",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default=None,
        help="path for folder where prediction images are to be saved. If none is given, no images will drawn",
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
        "--cuda", type=str, default="cuda:0", help="Cuda device string"
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
        "--crop-size",
        "-c",
        type=int,
        default=CROP_SIZE,
        help="Size for crops that will be predicted seperatly to prevent a cuda memory overflow",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--torch-seed", "-ts", type=float, default=314.0, help="Torch seed"
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        dest="scale",
        default=1,
        help="Downscaling factor of the images. Polygon data will be upscaled accordingly",
    )
    parser.add_argument(
        "--padding",
        "-p",
        dest="pad",
        type=int,
        nargs="+",
        default=FINAL_SIZE,
        help="Size to which the image will be padded to. Has to be a tuple (W, H). "
             "Has to be grater or equal to actual image",
    )
    parser.add_argument(
        "--bbox_threshold",
        "-bt",
        dest="bbox_size",
        type=int,
        default=500,
        help="Threshold for bboxes. Polygons, whose bboxes do not meet the requirement will be ignored. "
             "This will be adjusted depending on the scaling of the image.",
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
    return data


def predict(args: argparse.Namespace) -> None:
    """
    Loads all images from the data folder and predicts segmentation.
    """
    device = args.cuda if torch.cuda.is_available() else "cpu"
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

        pad = calculate_padding(args.pad, image.shape, args.scale)
        image = pad_image(pad, image)

        execute_prediction(args, device, file, image, model)


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
            pad[1] >= shape[1]
            and pad[0] >= shape[2]
    ), (
        f"Final size has to be greater than actual image size. "
        f"Padding to {pad[0]} x {pad[1]} "
        f"but image has shape of {shape[2]} x {shape[1]}"
    )

    pad = (pad[1] - shape[1], pad[0] - shape[2])
    return pad


def execute_prediction(args: argparse.Namespace, device: str, file: str, image: torch.Tensor, model: Any) -> None:
    """
    Run model to create prediction and call export methods. Todo: add switch for prediction with and cropping
    :param args:
    :param device:
    :param file:
    :param image:
    :param model:
    """
    # shape = (image.shape[1] // args.crop_size, image.shape[2] // args.crop_size)
    # crops = torch.tensor(Preprocessing.crop_img(args.crop_size, 1, np.array(image)))
    # predictions = []
    # dataloader = DataLoader(crops, batch_size=args.batch_size, shuffle=False)
    # for crop in dataloader:
    pred = torch.nn.functional.softmax(
        torch.squeeze(model(image[None, :].to(device)).detach().cpu()), dim=0
    ).numpy()
    # predictions.append(pred)

    # crops = torch.stack(predictions, dim=0)
    # crops = crops.permute(0, 2, 3, 1)
    # pred = torch.reshape(crops, (shape[0] * args.crop_size, shape[1] * args.crop_size, OUT_CHANNELS))
    # pred = pred.permute(2, 0, 1)

    pred = process_prediction(np.array(pred), args.threshold)
    if args.output_path:
        draw_prediction(pred, args.output_path + os.path.splitext(file)[0] + ".png")
    export_polygons(file, pred, args)


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


def export_polygons(file: str, pred: ndarray, args: argparse.Namespace) -> None:
    """
    Simplify prediction to polygons and export them to an image as well as transcribus xml
    :param args: arguments
    :param file: path
    :param pred: prediction 2d ndarray
    """
    if args.export:
        polygon_pred, reading_order_dict, segmentations = get_polygon_prediction(pred, int(args.bbox_size * args.scale))

        if args.output_path:
            draw_prediction(
                polygon_pred,
                args.output_path + f"{os.path.splitext(file)[0]}_polygons" + ".png",
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


def get_polygon_prediction(pred: ndarray, bbox_size: int) -> Tuple[
    ndarray, Dict[int, int], Dict[int, List[List[float]]]]:
    """
    Calls polyong conversion twice. Original segmentation is first converted to polygons, then those polygons are
    drawen into an ndarray image. This smothed prediction is again converted to polygons which are used to
    determine reading order.
    :param pred: Original prediction ndarray image
    :return: smothed prediction ndarray image, reading order and segmentation dictionary
    """
    segmentations, bbox_list = prediction_to_polygons(pred, TOLERANCE, bbox_size)
    polygon_pred = draw_polygons(segmentations, pred.shape)
    # segmentations, bbox_list = prediction_to_polygons(polygon_pred, TOLERANCE)

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
