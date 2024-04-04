"""Module for predicting newspaper images with trained models. """

import argparse
import os
from threading import Thread
from typing import Dict, List, Tuple, Any

# import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image
from numpy import ndarray
from skimage import draw
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from script.convert_xml import create_xml
from script.transkribus_export import prediction_to_polygons, get_reading_order
from src.news_seg import train  # pylint: disable=no-name-in-module
from src.news_seg.class_config import TOLERANCE
from src.news_seg.predict_dataset import PredictDataset
from src.news_seg.utils import create_bbox_ndarray, draw_prediction

# import train

DATA_PATH = "../../data/newspaper/input/"
RESULT_PATH = "../../data/output/"

CROP_SIZE = 1024
FINAL_SIZE = (1024, 1024)

# Tolerance pixel for polygon simplification. All points in the simplified object will be within
# the tolerance distance of the original geometry.


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
        "--slices-path",
        "-sp",
        type=str,
        default=None,
        help="path for folder where sclices are to be saved. If none is given, no slices will created",
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
    return parser.parse_args()


def collate_fn(batch: Any) -> Any:
    """dataloader collate function"""
    return (
        torch.stack([x[0] for x in batch]),
        [x[1] for x in batch]
    )


def predict(args: argparse.Namespace) -> None:
    """
    Loads all images from the data folder and predicts segmentation.
    Loading is handled through a Dataloader and Dataset. Threads are joined every 10 batches.
    """
    device = args.cuda if torch.cuda.is_available() else "cpu"
    cuda_count = torch.cuda.device_count()
    print(f"Using {device} device with {cuda_count} gpus")

    dataset = PredictDataset(args.data_path, args.scale, args.pad)

    pred_loader = DataLoader(
        dataset,
        batch_size=cuda_count,
        shuffle=False,
        num_workers=args.worker_factor * cuda_count,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn
    )
    if device != 'cpu':
        model = DataParallel(train.init_model(args.model_path, device, args.model_architecture, args.skip_cbam))
    else:
        model = train.init_model(args.model_path, device, args.model_architecture, args.skip_cbam)

    model.to(device)
    model.eval()
    threads = []

    batch = 0
    for image, path in tqdm(
            pred_loader, desc="predicting images", total=len(pred_loader), unit="batches"
    ):
        batch += 1
        threads += execute_prediction(args, device, path, image, model)

        if batch % 10 == 0:
            for thread in threads:
                thread.join()
            threads = []

    print("Prediction done, waiting for post processing to end")
    for thread in threads:
        thread.join()
    print("Done")


def execute_prediction(args: argparse.Namespace, device: str, paths: List[str], image: torch.Tensor,
                       model: Any) -> List[Thread]:
    """
    Run model to create prediction and start thrads for export.
    Todo: add switch for prediction with and without cropping
    :param args: arguments
    :param device: device, cuda or cpu
    :param paths: image file names
    :param image: image tensor [3, H, W]
    :param model: model to run prediction on
    """
    # shape = (image.shape[1] // args.crop_size, image.shape[2] // args.crop_size)
    # crops = torch.tensor(Preprocessing.crop_img(args.crop_size, 1, np.array(image)))
    # predictions = []
    # dataloader = DataLoader(crops, batch_size=args.batch_size, shuffle=False)
    # for crop in dataloader:
    with torch.autocast(device, enabled=args.amp):
        pred = torch.nn.functional.softmax(model(image.to(device)), dim=1)
    # predictions.append(pred)

    # crops = torch.stack(predictions, dim=0)
    # crops = crops.permute(0, 2, 3, 1)
    # pred = torch.reshape(crops, (shape[0] * args.crop_size, shape[1] * args.crop_size, OUT_CHANNELS))
    # pred = pred.permute(2, 0, 1)
    pred_ndarray = process_prediction(pred, args.threshold)
    image_ndarray = image.numpy()

    threads = []
    for i in range(pred_ndarray.shape[0]):
        threads.append(Thread(target=export_polygons, args=(paths[i], pred_ndarray[i], image_ndarray[i], args)))
        if args.output_path:
            draw_prediction(pred_ndarray[i], args.output_path + os.path.splitext(paths[i])[0] + ".png")
        threads[i].start()

    return threads


def export_polygons(file: str, pred: ndarray, image: ndarray, args: argparse.Namespace) -> None:
    """
    Simplify prediction to polygons and export them to an image as well as transcribus xml
    :param args: arguments
    :param file: path
    :param pred: prediction 2d ndarray
    """
    if args.export or args.slices_path or args.output_path:
        reading_order_dict, segmentations, bbox_list = polygon_prediction(pred, args)

        if args.slices_path:
            export_slices(args, file, image, reading_order_dict, segmentations, bbox_list, pred)

        if args.output_path:
            polygon_pred = draw_polygons_into_image(segmentations, pred.shape)
            draw_prediction(
                polygon_pred,
                args.output_path + f"{os.path.splitext(file)[0]}_polygons" + ".png",
            )
        if args.export:
            export_xml(args, file, reading_order_dict, segmentations)


def export_slices(args: argparse.Namespace, file: str, image: ndarray,
                  reading_order_dict: Dict[int, int], segmentations: Dict[int, List[List[float]]],
                  bbox_list: Dict[int, List[List[float]]], pred: ndarray) -> None:
    """
    Cuts slices out of the input image and applies mask. Those are being saved, sorted by input
    image and reading order on that nespaper page
    :param args: arguments
    :param file: file name
    :param image: input image (c, w, h)
    :param pred: prediction
    :param reading_order_dict: Dictionary for looking up reading order
    :param segmentations: polygons
    :param pred: prediction 2d ndarray uint8
    """
    if not os.path.exists(f"{args.slices_path}{os.path.splitext(file)[0]}"):
        os.makedirs(f"{args.slices_path}{os.path.splitext(file)[0]}")

    mask_list, reading_order_list, mask_bbox_list = get_slicing(segmentations, bbox_list,
                                                                reading_order_dict,
                                                                int(args.area_size * args.scale), pred)

    reading_order_dict = {k: v for v, k in enumerate(np.argsort(np.array(reading_order_list)))}
    for index, mask in enumerate(mask_list):
        bbox = mask_bbox_list[index]
        slice_image = image[:, int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2]), ]
        mean = np.mean(slice_image, where=mask == 0)
        slice_image = slice_image * mask
        slice_image = np.transpose(slice_image, (1, 2, 0))
        slice_image[slice_image[:, :, ] == (0, 0, 0)] = mean

        Image.fromarray((slice_image * 255).astype(np.uint8)).save(
            f"{args.slices_path}{os.path.splitext(file)[0]}/{reading_order_dict[index]}.png")


def export_xml(args: argparse.Namespace, file: str, reading_order_dict: Dict[int, int],
               segmentations: Dict[int, List[List[float]]]) -> None:
    """
    Open pre created transkribus xml files and save polygon xml data.
    :param args: args
    :param file: xml path
    :param reading_order_dict: reading order value for each index
    :param segmentations: polygon dictionary sorted by labels
    """
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


def polygon_prediction(pred: ndarray, args: argparse.Namespace) -> Tuple[
    Dict[int, int], Dict[int, List[List[float]]], Dict[int, List[List[float]]]]:
    """
    Calls polyong conversion. Original segmentation is first converted to polygons, then those polygons are
    drawen into an ndarray image. Furthermore, regions of sufficient size will be cut out and saved separately if
    required.
    :param args: args
    :param pred: Original prediction ndarray image
    :return: smothed prediction ndarray image, reading order and segmentation dictionary
    """
    segmentations, bbox_list = prediction_to_polygons(pred, TOLERANCE, int(args.bbox_size * args.scale),
                                                      args.export or args.output_path)

    bbox_ndarray = create_bbox_ndarray(bbox_list)
    reading_order: List[int] = []
    get_reading_order(bbox_ndarray, reading_order, int(args.separator_size * args.scale))
    reading_order_dict = {k: v for v, k in enumerate(reading_order)}

    return reading_order_dict, segmentations, bbox_list


def draw_polygons_into_image(
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


def area_sufficient(bbox: List[float], size: int) -> bool:
    """
    Calcaulates wether the area of the region is larger than parameter size.
    :param bbox: bbox list, minx, miny, maxx, maxy
    :param size: size to which the edges must at least sum to
    :return: bool value wether area is large enough
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > size


def get_slicing(
        segmentations: Dict[int, List[List[float]]], bbox_list: Dict[int, List[List[float]]],
        reading_order: Dict[int, int], area_size: int, pred: ndarray
) -> Tuple[List[ndarray], List[int], List[List[float]]]:
    """
    Takes segmentation dictionary and slices it in bbox pieces for each polygon
    :param reading_order: assings reading order position to each polygon index
    :param bbox_list: Dictionaray of bboxes sorted after label
    :param shape: shape of original image
    :param segmentations: dictionary assigning labels to polygon lists
    :return: result image as ndarray, reading order list and bbox list which correspond to the chosen regions
    """
    index = 0
    masks: List[ndarray] = []
    reading_order_list: List[int] = []
    mask_bbox_list: List[List[float]] = []
    for label, segmentation in segmentations.items():
        for key, _ in enumerate(segmentation):

            bbox = bbox_list[label][key]
            if area_sufficient(bbox, area_size):
                # polygon_ndarray = np.reshape(polygon, (-1, 2)).T
                # x_coords, y_coords = draw.polygon(polygon_ndarray[1], polygon_ndarray[0])
                create_mask(bbox, index, mask_bbox_list, masks, reading_order, reading_order_list, pred)
            index += 1
    return masks, reading_order_list, mask_bbox_list


def create_mask(bbox: List[float], index: int, mask_bbox_list: List[List[float]], masks: List[ndarray],
                reading_order: Dict[int, int], reading_order_list: List[int],
                pred: ndarray) -> None:
    """
    Draw mask into empyt image and cut out the bbox area. Masks, as well as reading order and bboxes are appended to
    their respective lists for further processing
    :param bbox:
    :param index:
    :param mask_bbox_list:
    :param masks:
    :param reading_order:
    :param reading_order_list:
    :param shape:
    :param x_coords:
    :param y_coords:
    """
    # temp_image = np.zeros(shape, dtype="uint8")
    # temp_image[x_coords, y_coords] = 1
    mask = pred[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]
    mask = (mask == 4).astype(np.uint8)
    masks.append(mask)
    reading_order_list.append(reading_order[index])
    mask_bbox_list.append(bbox)


def process_prediction(pred: torch.Tensor, threshold: float) -> ndarray:
    """
    Apply argmax to prediction and assign label 0 to all pixel that have a confidence below the threshold.
    :param threshold: confidence threshold for prediction
    :param pred: prediction [B, C, H, W]
    :return: prediction ndarray [B, H, W]
    """
    max_tensor, argmax = torch.max(pred, dim=1)
    argmax = argmax.type(torch.uint8)
    argmax[max_tensor < threshold] = 0
    return argmax.detach().cpu().numpy()  # type: ignore


if __name__ == "__main__":
    parameter_args = get_args()
    if not os.path.exists(f"{parameter_args.output_path}"):
        os.makedirs(f"{parameter_args.output_path}")
    if not os.path.exists(f"{parameter_args.slices_path}"):
        os.makedirs(f"{parameter_args.slices_path}")

    torch.manual_seed(parameter_args.torch_seed)
    predict(parameter_args)
