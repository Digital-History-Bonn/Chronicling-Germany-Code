import argparse
import glob
import json
import os
from os.path import basename, dirname
from typing import List, Literal

import torch
from bs4 import BeautifulSoup
from ultralytics import YOLO
from ultralytics.engine.results import Results

REGION_TYPES = {
    "caption": "TextRegion",
    "table": "TableRegion",
    "article": "TextRegion",
    "heading": "TextRegion",
    "header": "TextRegion",
    "image": "GraphicRegion",
    "inverted_text": "TextRegion",
}


def predict(model: YOLO, images: List[str], output_paths: List[str], device: str) -> None:
    """
    Predict with the model, write results to json or xml files.

    Args:
        model (YOLO): Yolo model to predict layout
        images: (List[str]): List of image paths
        output_paths (List[str]): List of xml file paths to write the results in
        device (str): device to use for prediction.
    """
    results = model.predict(images, device=device)

    if output_paths[0][-4:] == "json":
        write_jsons(output_paths, results)
    else:
        write_xmls(output_paths, results)


def write_xmls(output_paths: List[str], results: List[Results]) -> None:
    """
    Open pre created transkribus xml files and save polygon xml data.

    If xml files already exist, the regions are overwritten.
    Otherwise, it will be created from template.

    Args:
        output_paths (List[str]): paths to xml files
        results (List[Results]): List with results from prediction
    """
    # read template
    with open("src/cgprocess/layout_segmentation/templates/annotation_file.xml", 'r',
              encoding="utf-8") as f:
        template = f.read()

    for output_path, result in zip(output_paths, results):
        xml_data = BeautifulSoup(template, "xml")

        # create output path
        os.makedirs(dirname(output_path), exist_ok=True)

        page = xml_data.find("Page")
        page["imageFilename"] = basename(result.path)
        page["imageHeight"] = result.orig_shape[0]
        page["imageWidth"] = result.orig_shape[1]

        order_group = xml_data.new_tag(
            "OrderedGroup", attrs={"caption": "Regions reading order"}
        )

        for idx, (bbox, cls_idx) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
            bbox = bbox.int()
            cls_idx = cls_idx.item()

            order_group.append(
                xml_data.new_tag(
                    "RegionRefIndexed",
                    attrs={"index": str(idx), "regionRef": str(idx)},
                )
            )

            region = xml_data.new_tag(
                REGION_TYPES[result.names[cls_idx]],
                attrs={
                    "id": str(idx),
                    "custom": f"readingOrder {{index:{str(idx)};}} structure "
                              f"{{type:{result.names[cls_idx]};}}",
                },
            )
            region.append(
                xml_data.new_tag("Coords",
                                 attrs={"points": " ".join([f"{bbox[0]},{bbox[1]}",
                                                            f"{bbox[2]},{bbox[1]}",
                                                            f"{bbox[2]},{bbox[3]}",
                                                            f"{bbox[0]},{bbox[3]}"])}),
            )
            page.append(region)

        with open(output_path, "w", encoding="utf-8") as xml_file:
            xml_file.write(xml_data.prettify()) # type:ignore


def write_jsons(output_paths: List[str], results: List[Results]) -> None:
    """
    Writes result into a json file.

    Args:
        output_paths (List[str]): List of xml file path to write the results in.
        results (List[Results]): List with results form prediction
    """
    for result, output_path in zip(results, output_paths):
        data = {"bboxes": [],
                "scan_url": basename(result.path)}
        boxes = result.boxes.xyxy.cpu()
        classes = [result.names[cls.item()] for cls in result.boxes.cls.cpu()]
        data["bboxes"].extend([{"id": idx,
                                "bbox": {"x0": int(box[0]),
                                         "y0": int(box[1]),
                                         "x1": int(box[2]),
                                         "y1": int(box[3])},
                                "class": cls}
                               for idx, (box, cls) in enumerate(zip(boxes, classes))])

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


def main(image_path: str,
         output_path: str,
         model: str,
         file_format: Literal["json", "xml"]) -> None:
    """
    Predicts layout of the given images.

    Args:
        image_path (str): path to the folder with images
        output_path (str): path to the folder with xml files for results
        model (str): path to the model
        file_format: format of the output (json or xml)
    """
    # load model
    model = YOLO(model) # type: ignore

    # create list of images
    images = list(glob.glob(f"{image_path}/*.jpg"))

    # create folder for output and create list of paths
    os.makedirs(output_path, exist_ok=True)
    output_paths = [f"{output_path}/{basename(x)[:-4]}.{file_format}" for x in images]

    # get gpus
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    print(f"using {device} for prediction")

    # predict
    predict(model, images, output_paths, device=device) # type: ignore


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="predict")
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--images",
        "-i",
        type=str,
        default="data/YOLO_dataset/test/images",
        help="path the the images to predict"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="predictions/layout/YOLO/page",
        help="Number of epochs to train."
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="YOLO_detect_2",
        help="Name of the model to use for prediction."
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="xml",
        help="Format to save the prediction in (json or xml)."
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args.images, args.output, args.model, args.format)
