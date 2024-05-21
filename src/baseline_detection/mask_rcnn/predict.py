"""Prediction script for Mask R-CNN baseline detection."""

from pathlib import Path
from typing import List, Tuple

import torch
from shapely import Polygon, LineString
from shapely.affinity import translate
from skimage.measure import find_contours
from torchvision.transforms import GaussianBlur
from skimage import io
from torchvision.models.detection import MaskRCNN

from src.baseline_detection.mask_rcnn.postprocessing import postprocess
from src.baseline_detection.mask_rcnn.preprocess import extract
from src.baseline_detection.mask_rcnn.trainer_textline import get_model
from monai.networks.nets import BasicUNet

from src.baseline_detection.xml_conversion import add_baselines


def prior(size: int) -> torch.Tensor:
    """
    Creates a prior map for the baseline in the bounding box.

    Args:
        size (int): Height of the bounding box

    Returns:
        tensor of size with vertical prior values
    """
    p1 = int(size * 0.3)
    p2 = int(size * 0.6)
    p3 = int(size * 0.80)
    return torch.hstack([torch.zeros(p1),
                         torch.linspace(0.0, 1, p2 - p1),
                         torch.ones(p3 - p2),
                         torch.linspace(1, 0.0, size - p3)])


def predict_baseline(box: torch.Tensor, mask: torch.Tensor, map: torch.Tensor) -> LineString:
    """
    Predicts baseline in bounding box using the mask and probability map.

    Args:
        box (torch.Tensor): Bounding box of textline
        mask (torch.Tensor): Mask of textline
        map (torch.Tensor): Probability map of baseline inside bounding box

    Returns:
        LineString of baseline in bounding box
    """
    line_region = map * mask[0]
    line_region = line_region[box[1]:box[3], box[0]:box[2]]
    line_region = prior((box[3] - box[1]).item())[:, None] * line_region     # type: ignore

    y_pos = torch.argmax(line_region, dim=0)
    y_values = torch.amax(line_region, dim=0)

    line = torch.vstack([box[1] + y_pos, torch.arange(box[0], box[0] + len(y_pos))])  # type: ignore
    line = line.T[y_values > 0.5]

    return LineString(line.int()).simplify(tolerance=1)


def get_polygon(mask: torch.Tensor) -> Polygon:
    """
    Converts mask into one polygon representing the textline.

    Args:
        mask (torch.Tensor): Mask of textline

    Returns:
        Polygon of textline
    """
    polygons = find_contours(mask)
    lengths = torch.tensor([len(polygon) for polygon in polygons])
    idx = torch.argmax(lengths)
    return Polygon(polygons[idx]).simplify(tolerance=1)


def predict_image(textline_model: MaskRCNN,
                  baseline_model: BasicUNet,
                  image: torch.Tensor,
                  device: torch.device) -> Tuple[List[Polygon], List[LineString]]:
    """
    Predicts textlines and baselines in given image.

    Args:
        textline_model (MaskRCNN): Mask R-CNN model for textline detection
        baseline_model (BasicUNet): Mask R-CNN model for baseline prediction
        image (torch.Tensor): Image of paragraph of page
        device (torch.device): Device used for prediction

    Returns:
        textline and baseline predictions
    """
    gauss_filter = GaussianBlur(kernel_size=(25, 3), sigma=5.0)
    image = image.to(device)

    # predict example form training set
    pred = textline_model([image])[0]

    del pred['labels']

    # move predictions to cpu
    pred["boxes"] = pred["boxes"].detach().cpu()
    pred["scores"] = pred["scores"].detach().cpu()
    pred["masks"] = pred["masks"].detach().cpu()

    # postprecess image (non maxima supression)
    pred = postprocess(pred, method='iom', threshold=.6)

    # baseline predictions
    baseline_probability_map = baseline_model(image[None])[0, 1]
    baseline_probability_map = baseline_probability_map.detach().cpu()
    baseline_probability_map = gauss_filter(baseline_probability_map[None])[0]

    baselines = []
    for box, mask in zip(pred["boxes"], pred["masks"]):
        baselines.append(predict_baseline(box, mask, baseline_probability_map))

    return [get_polygon(mask[0].numpy()) for mask in pred['masks']], baselines


def predict_page(image: torch.Tensor,
                 bounding_boxes: List[torch.Tensor],
                 reading_order: List[int]
                 ) -> Tuple[List[Polygon], List[LineString]]:
    """
    Predicts textlines and baselines in given image (complete page).

    Args:
        image: Image of complete page
        bounding_boxes: bounding boxes of textregions form layout segmentation
        reading_order: Reading order of textregions

    Returns:
        Prediction dict with boundingboxes, probability score, textlines,
        baselines, region and readingOrder.
    """
    # set device
    device = torch.device('cuda:0')

    # init and load model for textline detection
    textline_model = get_model(load_weights='MaskRCCNLineDetection2_Newspaper_textlines_e25_es')
    textline_model.to(device)
    textline_model.eval()

    # init and load model for baseline detection
    baseline_model = BasicUNet(spatial_dims=2, in_channels=3, out_channels=2)
    baseline_model.load_state_dict(
        torch.load(f'{Path(__file__).parent.absolute()}/../../../models/test3_baseline_e100_es.pt'))
    baseline_model.to(device)
    baseline_model.eval()

    textlines = []
    baselines = []

    # iterate over regions and predict lines
    for _, box in sorted(zip(reading_order, bounding_boxes)):
        pred_textlines, pred_baselines = predict_image(textline_model,
                                                       baseline_model,
                                                       image[:, box[0]: box[2], box[1]: box[3]],
                                                       device)

        baselines.extend([translate(line, xoff=box[0], yoff=box[1])
                         for line in pred_baselines])
        textlines.extend([translate(mask, xoff=box[0], yoff=box[1])
                         for mask in pred_textlines])

    return textlines, baselines


def main(image_path: str, layout_xml_path: str, output_file: str) -> None:
    """
    Predicts textlines and baselines in given image and writes into a annotation xml file.

    Args:
        image_path (str): Path to image
        layout_xml_path (str): Path to layout xml file
        output_file (str): Path to output xml file
    """
    image = torch.tensor(io.imread(image_path)).permute(2, 0, 1)
    image = image.to(torch.device('cuda:0'))
    image = image.float()

    annotations, _ = extract(layout_xml_path)
    textlines, baselines = predict_page(image,
                                        [a['part'] for a in annotations],           # type: ignore
                                        [a['readingOrder'] for a in annotations])   # type: ignore

    add_baselines(
        layout_xml_path,
        output_file,
        textlines,
        baselines
    )


if __name__ == '__main__':
    main(
        image_path=f"{Path(__file__).parent.absolute()}/../../../"
                   f"data/images/Koelnische_Zeitung_1924 - 0085.jpg",
        layout_xml_path=f"{Path(__file__).parent.absolute()}/../../../"
                        f"data/pero_lines_bonn_regions/Koelnische_Zeitung_1924 - 0085.xml",
        output_file=f"{Path(__file__).parent.absolute()}/../../../data/predictionExample.xml"
    )
