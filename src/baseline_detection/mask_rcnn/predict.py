"""Prediction script for Mask R-CNN baseline detection."""

from pathlib import Path
from typing import Dict, Union, List

import torch
from shapely import Polygon, LineString, Geometry
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


def prior(size: int):
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
    line_region = line_region[box[1]:box[3], box[0]:box[2]] * prior(box[3] - box[1])[:, None]

    y_pos = torch.argmax(line_region, dim=0)
    y_values = torch.amax(line_region, dim=0)

    line = torch.vstack([box[1] + y_pos,
                         torch.arange(box[0], box[0] + len(y_pos))]).T[y_values > 0.5]

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
                  device: torch.device) -> Dict[str, Union[torch.Tensor, Geometry]]:
    """
    Predicts textlines and baselines in given image.

    Args:
        textline_model (MaskRCNN): Mask R-CNN model for textline detection
        baseline_model (BasicUNet): Mask R-CNN model for baseline prediction
        image (torch.Tensor): Image of paragraph of page
        device (torch.device): Device used for prediction

    Returns:
        Dict with predictions for:
            boxes: textline bounding boxes
            scores: probabilities of textlines
            textlines: polygons of textlines
            baselines: Linestrings of textlines
    """
    gauss_filter = GaussianBlur(kernel_size=5, sigma=2.0)
    image = image.to(device)

    # predict example form training set
    pred: Dict[str, Union[torch.Tensor,
                          List[torch.Tensor],
                          List[Polygon],
                          List[LineString]]] = textline_model([image])[0]

    # move predictions to cpu
    pred["boxes"] = pred["boxes"].detach().cpu()
    pred["labels"] = pred["labels"].detach().cpu()
    pred["scores"] = pred["scores"].detach().cpu()
    pred["masks"] = pred["masks"].detach().cpu()

    # postprecess image (non maxima supression)
    pred = postprocess(pred, method='iom', threshold=.6)

    baseline_probability_map = baseline_model(image[None])[0, 1]

    baseline_probability_map = baseline_probability_map.detach().cpu()
    baseline_probability_map = gauss_filter(baseline_probability_map[None])[0]

    pred['lines'] = []
    for box, mask in zip(pred["boxes"], pred["masks"]):
        pred['lines'].append(predict_baseline(box, mask, baseline_probability_map))

    pred['textline'] = [get_polygon(mask[0].numpy()) for mask in pred['masks']]
    del pred['masks']
    del pred['labels']

    return pred


def predict_page(image: torch.Tensor, annotation: List[Dict[str, List[torch.Tensor]]]):
    """
    Predicts textlines and baselines in given image (complete page).

    Args:
        image (torch.Tensor): Image of complete page
        annotation (List[Dict[str, List[torch.Tensor]]]): annotations of page layout,
                                                          containing TextRegions

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

    prediction = {"boxes": torch.zeros((0, 4)),
                  "scores": torch.zeros((0,)),
                  "textline": [],
                  "lines": [],
                  "region": [],
                  "readingOrder": []}

    # iterate over regions and predict lines
    for region in sorted(annotation, key=lambda anno: anno['readingOrder']):
        subimage = image[:, region['part'][0]: region['part'][2],
                         region['part'][1]: region['part'][3]]

        pred = predict_image(textline_model, baseline_model, subimage, device)

        shift = torch.tensor([region['part'][1], region['part'][0],
                              region['part'][1], region['part'][0]])
        prediction["boxes"] = torch.vstack([prediction["boxes"], pred["boxes"] + shift])
        prediction["scores"] = torch.hstack([prediction["scores"], pred["scores"]])

        length = len(pred['lines'])
        shift = torch.tensor([region['part'][0], region['part'][1]])
        prediction["lines"].extend([translate(line, xoff=shift[0], yoff=shift[1])
                                    for line in pred["lines"]])
        prediction["textline"].extend([translate(mask, xoff=shift[0], yoff=shift[1])
                                       for mask in pred["textline"]])
        prediction["region"].extend([region['readingOrder']] * length)
        prediction["readingOrder"].extend([i for i in range(length)])

    return prediction


def main(image_path: str, layout_xml_path: str, output_file: str):
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

    anno, _ = extract(layout_xml_path)
    pred = predict_page(image, anno)

    add_baselines(
        layout_xml_path,
        output_file,
        pred['textline'],
        pred['lines']
    )


if __name__ == '__main__':
    main(
        image_path=f"{Path(__file__).parent.absolute()}/../../../"
                   f"data/images/Koelnische_Zeitung_1924 - 0085.jpg",
        layout_xml_path=f"{Path(__file__).parent.absolute()}/../../../"
                        f"data/pero_lines_bonn_regions/Koelnische_Zeitung_1924 - 0085.xml",
        output_file=f"{Path(__file__).parent.absolute()}/../../../data/predictionExample.xml"
    )
