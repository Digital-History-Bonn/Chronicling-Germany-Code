"""Module for predicting newspaper images with trained models. """

import argparse
import os
import warnings
from threading import Thread
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from numpy import ndarray
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cgprocess.layout_segmentation.class_config import TOLERANCE
from src.cgprocess.layout_segmentation.datasets.predict_dataset import PredictDataset
from src.cgprocess.layout_segmentation.helper.train_helper import init_model
from src.cgprocess.layout_segmentation.processing.draw_img_from_polygons import draw_polygons_into_image
from src.cgprocess.layout_segmentation.processing.polygon_handler import prediction_to_region_polygons, uncertainty_to_polygons
from src.cgprocess.layout_segmentation.processing.reading_order import PageProperties
from src.cgprocess.layout_segmentation.processing.slicing_export import export_slices
from src.cgprocess.layout_segmentation.processing.transkribus_export import export_xml
from src.cgprocess.layout_segmentation.train_config import OUT_CHANNELS
from src.cgprocess.layout_segmentation.utils import draw_prediction, adjust_path, collapse_prediction

DATA_PATH = "../../../data/newspaper/input/"
RESULT_PATH = "../../../data/output/"

CROP_SIZE = 1024
FINAL_SIZE = (1024, 1024)


# Tolerance pixel for polygon simplification. All points in the simplified object will be within
# the tolerance distance of the original geometry.


def predict_batch(args: argparse.Namespace, device: str, paths: List[str], image: torch.Tensor,
                  target: torch.Tensor, model: Any, debug: bool = False) -> List[Thread]:
    """
    Run model to create prediction of whole batch and start export threads for each image.
    :param debug: if true, the uncertainty prediction is activated
    :param target: target for uncertainty prediction.
    :param args: arguments
    :param device: device, cuda or cpu
    :param paths: image file names
    :param image: image tensor [3, H, W]
    :param model: model to run prediction on
    """
    with torch.autocast(device, enabled=args.amp):
        pred = torch.nn.functional.softmax(model(image.to(device)), dim=1)

    if debug:
        pred_ndarray = process_prediction_debug(pred, target.to(device), args.threshold)
    else:
        pred_ndarray = process_prediction(pred, args.threshold, args.reduce)
    image_ndarray = image.numpy()

    threads = []
    for i in range(pred_ndarray.shape[0]):
        threads.append(Thread(target=image_level_export, args=(paths[i], pred_ndarray[i], image_ndarray[i], args)))
        if args.output_path:
            draw_prediction(pred_ndarray[i], adjust_path(args.output_path) + os.path.splitext(paths[i])[0] + ".png")
        threads[i].start()

    return threads


def image_level_export(file: str, pred: ndarray, image: ndarray, args: argparse.Namespace) -> None:
    """
    Handle export options.
    :param args: arguments
    :param file: path
    :param pred: prediction 2d ndarray
    """
    if args.export or args.slices_path or args.output_path:
        reading_order_dict, segmentations, bbox_list = get_region_polygons(pred, args)

        if args.slices_path:
            export_slices(args, file, image, reading_order_dict, segmentations, bbox_list, pred)

        if args.output_path:
            np.save(args.output_path + f"{os.path.splitext(file)[0]}_polygons" + ".npy", pred)
            polygon_pred = draw_polygons_into_image(segmentations, pred.shape)
            draw_prediction(
                polygon_pred,
                adjust_path(args.output_path) + f"{os.path.splitext(file)[0]}_polygons" + ".png",
            )
        if args.export:
            export_xml(args, file, reading_order_dict, segmentations, image.shape)


def get_region_polygons(pred: ndarray, args: argparse.Namespace) -> Tuple[
    Dict[int, int], Dict[int, List[List[float]]], Dict[int, List[List[float]]]]:
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
        segmentations, bbox_list = prediction_to_region_polygons(pred, TOLERANCE, int(args.bbox_size * args.scale),
                                                                 args.export or args.output_path)
        page = PageProperties(bbox_list)
        reading_order_dict = page.get_reading_order()

    return reading_order_dict, segmentations, bbox_list


def process_prediction(pred: torch.Tensor, threshold: float, reduce: bool = False) -> ndarray:
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


def process_prediction_debug(prediction: torch.Tensor, target: torch.Tensor, threshold: float) -> np.ndarray:
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
    parser.add_argument(
        "--cuda", type=str, default="cuda", help="Cuda device string"
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
        "--worker-factor",
        "-wf",
        type=int,
        default=2,
        help="Factor for number of workers. There will be gpu_count * worker_factor many Factor.",
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
    return parser.parse_args()


def collate_fn(batch: Any) -> Any:
    """dataloader collate function"""
    return (
        torch.stack([x[0] for x in batch]),
        torch.stack([x[1] for x in batch]),
        [x[2] for x in batch]
    )


def predict(args: argparse.Namespace) -> None:
    """
    Loads all images from the data folder and predicts segmentation.
    Loading is handled through a Dataloader and Dataset. Threads are joined every 10 batches.
    """
    device = args.cuda if torch.cuda.is_available() else "cpu"
    cuda_count = torch.cuda.device_count()

    target_path = adjust_path(args.target_path if args.uncertainty_predict else None)
    dataset = PredictDataset(adjust_path(args.data_path),
                             args.scale, args.pad,
                             target_path=adjust_path(target_path))

    print(f"{len(dataset)=}")

    pred_loader = DataLoader(
        dataset,
        batch_size=cuda_count if torch.cuda.is_available() else 1,
        shuffle=False,
        num_workers=args.worker_factor * cuda_count,
        prefetch_factor=2 if args.worker_factor * cuda_count > 1 else None,
        persistent_workers=args.worker_factor * cuda_count > 1,
        pin_memory=True,
        collate_fn=collate_fn
    )
    if device == 'cpu':
        print(f"Using {device}")
        model = init_model(args.model_path, device, args.model_architecture, args.skip_cbam,
                           overwrite_load_channels=args.override_load_channels)
    else:
        print(f"Using {device} device with {cuda_count} gpus")
        model = DataParallel(
            init_model(args.model_path, device, args.model_architecture, args.skip_cbam, args.override_load_channels))

    model.to(device)
    model.eval()
    threads = []

    batch = 0
    for image, target, path in tqdm(
            pred_loader, desc="layout inference", total=len(pred_loader), unit="batches"
    ):
        batch += 1
        threads += predict_batch(args, device, path, image, target, model, args.uncertainty_predict)

        if batch % 10 == 0:
            for thread in threads:
                thread.join()
            threads = []

    print("Prediction done, waiting for post processing to end")
    for thread in threads:
        thread.join()
    print("Done")


if __name__ == "__main__":
    parameter_args = get_args()

    if parameter_args.output_path:
        warnings.warn("Image output slows down the prediction significantly. "
                      "--output-path should not be activated in production environment.")

    if not os.path.exists(f"{parameter_args.output_path}"):
        os.makedirs(f"{parameter_args.output_path}")
    if not os.path.exists(f"{parameter_args.slices_path}"):
        os.makedirs(f"{parameter_args.slices_path}")

    torch.manual_seed(parameter_args.torch_seed)
    predict(parameter_args)
