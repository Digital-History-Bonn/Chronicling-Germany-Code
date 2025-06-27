"""Module for predicting newspaper images with trained models."""

import argparse
import os
import warnings
from multiprocessing import set_start_method
from threading import Thread
from typing import Dict, List, Tuple

import numpy as np
import torch
from numpy import ndarray
from torch import nn

from cgprocess.layout_segmentation.class_config import TOLERANCE
from cgprocess.layout_segmentation.datasets.predict_dataset import PredictDataset
from cgprocess.layout_segmentation.helper.train_helper import init_model
from cgprocess.layout_segmentation.processing.draw_img_from_polygons import (
    draw_polygons_into_image,
)
from cgprocess.layout_segmentation.processing.polygon_handler import (
    prediction_to_region_polygons,
    uncertainty_to_polygons,
)
from cgprocess.layout_segmentation.processing.reading_order import PageProperties
from src.cgprocess.layout_segmentation.processing.slicing_export import export_slices
from src.cgprocess.layout_segmentation.processing.transkribus_export import export_xml
from src.cgprocess.layout_segmentation.train_config import OUT_CHANNELS
from src.cgprocess.layout_segmentation.utils import (
    adjust_path,
    collapse_prediction,
    create_model_list,
    create_path_queue,
    draw_prediction,
)
from src.cgprocess.shared.multiprocessing_handler import MPPredictor

DATA_PATH = "../../../data/newspaper/input/"
RESULT_PATH = "../../../data/output/"

CROP_SIZE = 1024
FINAL_SIZE = (1024, 1024)


# Tolerance pixel for polygon simplification. All points in the simplified object will be within
# the tolerance distance of the original geometry.


def predict(args: list, model: nn.Module) -> Thread:
    """
    Run model to create prediction of one image and start export thread.
    :param args: arguments passed from multiprocessing handler.
    :param model: layout model for prediction.
    """
    path, cmd_args, dataset, _ = args

    target_path = adjust_path(
        cmd_args.target_path if cmd_args.uncertainty_predict else None
    )
    model.eval()
    device = next(model.parameters()).device
    debug = target_path is not None

    image, target = dataset.load_data_by_path(path)
    image = image[None, :, :, :]

    pred = torch.nn.functional.softmax(model(image.to(device)), dim=1)

    if debug:
        target = target[None, :, :]  # type: ignore
        pred_ndarray = process_prediction_debug(
            pred, target.to(device), cmd_args.threshold
        )
    else:
        pred_ndarray = process_prediction(pred, cmd_args.threshold, cmd_args.reduce)
    image_ndarray = image.numpy()

    # todo: thread exceptions must be caught and image name added to failed queue
    thread = Thread(
        target=image_level_export,
        args=(path, pred_ndarray[0], image_ndarray[0], cmd_args),
    )
    thread.start()
    if cmd_args.output_path:
        draw_prediction(
            pred_ndarray[0],
            adjust_path(cmd_args.output_path) + os.path.splitext(path)[0] + ".png",
        )

    return thread


def image_level_export(
    file: str, pred: ndarray, image: ndarray, args: argparse.Namespace
) -> None:
    """
    Handle export options.
    :param args: arguments
    :param file: path
    :param pred: prediction 2d ndarray
    """
    if args.export or args.slices_path or args.output_path:
        reading_order_dict, segmentations, bbox_list = get_region_polygons(pred, args)

        if args.slices_path:
            export_slices(
                args, file, image, reading_order_dict, segmentations, bbox_list, pred
            )

        if args.output_path:
            np.save(
                args.output_path + f"{os.path.splitext(file)[0]}_polygons" + ".npy",
                pred,
            )
            polygon_pred = draw_polygons_into_image(segmentations, pred.shape)
            draw_prediction(
                polygon_pred,
                adjust_path(args.output_path)
                + f"{os.path.splitext(file)[0]}_polygons"
                + ".png",
            )
        if args.export:
            export_xml(args, file, reading_order_dict, segmentations, image.shape)


def get_region_polygons(
    pred: ndarray, args: argparse.Namespace
) -> Tuple[Dict[int, int], Dict[int, List[List[float]]], Dict[int, List[List[float]]]]:
    """
    Calls polygon conversion. Original segmentation is first converted to polygons, then those polygons are
    drawn into a ndarray image. Furthermore, regions of sufficient size will be cut out and saved separately if
    required.
    :param args: args
    :param pred: Original prediction ndarray image
    :return: smoothed prediction ndarray image, reading order and segmentation dictionary
    """

    if args.uncertainty_predict:
        segmentations, bbox_list = uncertainty_to_polygons(pred)
        reading_order_dict = {i: i for i in range(len(segmentations))}

    else:
        segmentations, bbox_list = prediction_to_region_polygons(
            pred,
            TOLERANCE,
            int(args.bbox_size * args.scale),
            args.export or args.output_path,
        )
        page = PageProperties(bbox_list)
        reading_order_dict = page.get_reading_order()

    return reading_order_dict, segmentations, bbox_list


def process_prediction(
    pred: torch.Tensor, threshold: float, reduce: bool = False
) -> ndarray:
    """
    Apply argmax to prediction and assign label 0 to all pixel that have a confidence below the threshold.
    :param reduce: if true, this will load the reduce dictionary from class_config.py to combine probability values
    from specified classes. For example, heading and paragraphs are combined to paragraphs.
    :param threshold: confidence threshold for prediction
    :param pred: prediction [B, C, H, W]
    :return: prediction ndarray [B, H, W]
    """
    if reduce:
        pred = collapse_prediction(pred)

    max_tensor, argmax = torch.max(pred, dim=1)
    argmax = argmax.type(torch.uint8)
    argmax[max_tensor < threshold] = 0
    return argmax.detach().cpu().numpy()  # type: ignore


def process_prediction_debug(
    prediction: torch.Tensor, target: torch.Tensor, threshold: float
) -> np.ndarray:
    """
    Extract uncertain predictions based on the ground truth. Uncertain are all pixel with a predicted probability
    for the target class below a given threshold
    :param prediction: prediction from model
    :param target: ground truth
    :param threshold: limit for model confidence to consider prediction as certain
    :return numpy array with uncertain pixels [B, H, W]
    """
    confidence_for_target = torch.gather(prediction, 1, target.long())
    uncertainty_map: torch.Tensor = confidence_for_target < threshold

    # create a numpy array with class 1 at uncertain pixels
    mask = uncertainty_map.detach().cpu().numpy()
    uncertainty = np.zeros_like(mask, dtype=np.uint8)
    uncertainty[mask] = 1

    return uncertainty[:, 0, :, :]


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="Newspaper Layout Prediction")
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default=DATA_PATH,
        help="Path for folder with images to be segmented. Images need to be png or jpg. Otherwise they"
        " will be skipped",
    )
    parser.add_argument(
        "--target-path",
        "-tp",
        type=str,
        default="targets/",
        help="Path for folder with targets for uncertainty prediction. Have to be .npy files.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default=None,
        help="path for folder where prediction images are to be saved. If none is given, no images will drawn",
    )
    parser.add_argument(
        "--slices-path",
        "-sp",
        type=str,
        default=None,
        help="path for folder where slices are to be saved. If none is given, no slices will created",
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
    parser.add_argument("--cuda", type=str, default="cuda", help="Cuda device string")
    parser.add_argument(
        "--threshold",
        "-th",
        type=float,
        default=0.5,
        help="Confidence threshold for assigning a label to a pixel.",
    )
    parser.add_argument(
        "--model-architecture",
        "-a",
        type=str,
        default="dh_segment",
        help="which model to load options are 'dh_segment, trans_unet, dh_segment_small, dh_segment_2, dh_segment_wide",
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
        "--bbox-threshold",
        "-bt",
        dest="bbox_size",
        type=int,
        default=500,
        help="Threshold for bboxes. Polygons, whose bboxes do not meet the requirement will be ignored. "
        "This will be adjusted depending on the scaling of the image.",
    )
    parser.add_argument(
        "--skip-cbam",
        action="store_true",
        help="Activates cbam skip connection. Does only have an effect if the cbam dhsegment model is used",
    )
    parser.add_argument(
        "--separator-threshold",
        "-st",
        dest="separator_size",
        type=int,
        default=1000,
        help="Threshold for big separators. Only big separators that meet the requirement are valid to "
        "split reading order. This will be adjusted depending on the scaling of the image.",
    )
    parser.add_argument(
        "--area-threshold",
        "-at",
        dest="area_size",
        type=int,
        default=800000,
        help="Threshold for Regions that are large enough to contain a lot of text and will be cut out "
        "for further processing",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Activates automated mixed precision",
    )
    parser.add_argument(
        "--reduce",
        action="store_true",
        help="Activates class reduction after softmax.",
    )
    parser.add_argument(
        "--uncertainty-predict",
        action="store_true",
        help="Activates the uncertainty prediction. This writes regions into the xml file, which correspond to false "
        "classified regions. Class 1 (caption) = false class 2 (table) = true.",
    )
    parser.add_argument(
        "--override-load-channels",
        type=int,
        default=OUT_CHANNELS,
        help="This overrides the number of classes, with that a model will be loaded. The pretrained model will be "
        "loaded with this number of output classes instead of the configured number. This is necessary if a "
        "pretrained model is intended to be used for a task with a different number of output classes.",
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=1,
        help="Number of processes for every gpu in use. This has to be used carefull, as this can lead to a CUDA out "
        "of memory error.",
    )
    parser.add_argument(
        "--thread-count",
        "-t",
        type=int,
        default=1,
        help="Select number of threads that are launched per process.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Loads all images from the data folder and predicts segmentation.
    Loading is handled through a Dataloader and Dataset. Threads are joined every 10 batches.
    """

    target_path = adjust_path(args.target_path if args.uncertainty_predict else None)
    data_path = adjust_path(args.data_path)
    dataset = PredictDataset(
        data_path, args.scale, target_path=adjust_path(target_path)  # type: ignore
    )

    print(f"{len(dataset)=}")

    num_gpus = torch.cuda.device_count()
    num_processes = args.processes

    # create queue
    path_queue = create_path_queue(dataset.file_names, args, dataset)

    model_list = create_model_list(args, num_gpus, num_processes)

    predictor = MPPredictor(
        "Layout prediction",
        predict,
        init_model,
        path_queue,
        model_list,
        data_path,
        False,
        False,
    )  # type: ignore
    predictor.launch_processes(num_gpus, args.thread_count)


if __name__ == "__main__":
    set_start_method("spawn")
    parameter_args = get_args()

    if parameter_args.output_path:
        warnings.warn(
            "Image output slows down the prediction significantly. "
            "--output-path should not be activated in production environment."
        )

    if not os.path.exists(f"{parameter_args.output_path}"):
        os.makedirs(f"{parameter_args.output_path}")
    if not os.path.exists(f"{parameter_args.slices_path}"):
        os.makedirs(f"{parameter_args.slices_path}")

    torch.manual_seed(parameter_args.torch_seed)
    main(parameter_args)
