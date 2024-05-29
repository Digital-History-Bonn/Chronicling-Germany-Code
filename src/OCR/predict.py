"""OCR prediction module."""

import argparse
import glob
import multiprocessing
import os
from typing import Tuple, List

import torch
from PIL import Image
from bs4 import BeautifulSoup, PageElement
from kraken import rpred
from kraken.containers import Segmentation, BaselineLine    # pylint: disable=no-name-in-module, import-error
from kraken.lib import models
from kraken.lib.models import TorchSeqRecognizer
from tqdm import tqdm

from src.OCR.utils import pad_xml, pad_image


def extract_baselines(anno_path: str) -> Tuple[BeautifulSoup,
                                               List[PageElement],
                                               List[List[BaselineLine]]]:
    """
    Extracts and pads baselines from xml file for prediction.

    Args:
        anno_path: Path to annotation file.

    Returns:
        BeautifulSoup element with removed old text if exists,
        List of Textregions,
        List with BaselineLine Objects used for prediction
    """
    # load and pad annotations
    with open(anno_path, 'r', encoding='utf-8') as file:
        xml_data = file.read()

    soup = BeautifulSoup(xml_data, 'xml')
    soup = pad_xml(soup)

    # remove all textEquiv elements from annotations
    textequiv_objects = soup.find_all('TextEquiv')
    for textequiv_object in textequiv_objects:
        textequiv_object.decompose()

    # init baseline and ground
    baselines: List[List[BaselineLine]] = []

    # Find all text regions
    text_regions = soup.find_all('TextRegion')

    for region in text_regions:
        region_baselines: List[BaselineLine] = []

        # Find all TextLine elements
        text_lines = region.find_all('TextLine')

        # Extract Baseline points from each TextLine
        for i, text_line in enumerate(text_lines):
            polygon = text_line.find('Coords')
            baseline = text_line.find('Baseline')
            if baseline:
                # Split the points string and convert them to a list of tuples
                region_baselines.append(
                    BaselineLine(
                        str(i),
                        [tuple(map(int, point.split(','))) for point in
                         baseline['points'].split()],  # type: ignore
                        [tuple(map(int, point.split(','))) for point in
                         polygon['points'].split()])  # type: ignore
                )

        baselines.append(region_baselines)

    return soup, text_regions, baselines


def predict(model: TorchSeqRecognizer, image_path: str, anno_path: str, out_path: str) -> None:
    """
    Predicts OCR on given image using given baseline annotations and model.

    Args:
        model: Model to use for OCR prediction.
        image_path: Path to image file.
        anno_path: Path to annotation file.
        out_path: Path to output directory.
    """
    print(f'Predicting {image_path}...')
    # load image and pad image
    im = pad_image(Image.open(image_path))
    file_name = os.path.basename(image_path)

    # preprocess annotations and extract baselines
    soup, regions, region_baselines = extract_baselines(anno_path)

    for region, baselines in zip(regions, region_baselines):
        baseline_seg = Segmentation(type='baselines',
                                    imagename=file_name,
                                    text_direction='horizontal-lr',
                                    script_detection=False,
                                    lines=baselines,
                                    line_orders=[])

        # plotline = [np.array([b.baseline]).reshape(-1, 2) for b in baselines]
        # plotpolygon = [np.array([b.boundary]).reshape(-1, 2) for b in baselines]
        # plot_boxes_on_image(im, plotline, plotpolygon, f"baselines_{i}")

        # single model recognition
        pred_it = rpred.rpred(model, im, baseline_seg)
        lines = list(pred_it)

        textlines = region.find_all('TextLine')
        for pred_line, textline in zip(lines, textlines):
            textequiv = soup.new_tag('TextEquiv')
            if pred_line.confidences:
                textequiv['conf'] = str(min(pred_line.confidences))
            else:
                textequiv['conf'] = '0'
            unicode = soup.new_tag('Unicode')
            unicode.string = str(pred_line).strip()
            textequiv.append(unicode)
            textline.append(textequiv)

    # unpad coordinates in annotation by invert padding
    soup = pad_xml(soup, pad_value=-10)

    # save results
    with open(out_path, 'w', encoding='utf-8') as file:
        file.write(soup.prettify()
                   .replace("<Unicode>\n      ", "<Unicode>")
                   .replace("\n     </Unicode>", "</Unicode>"))


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="predict")
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="path for folder with images and xml files with baselines. Images need to be jpg."
    )
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--layout_dir",
        "-l",
        type=str,
        default=None,
        help="path for folder with layout xml files."
    )
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="path to the folder where the xml files with the prediction are saved",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="models/ocr_model.mlmodel",
        help="path to model .mlmodel file",
    )

    parser.add_argument(
        "--cuda",
        "-c",
        type=int,
        default=0,
        help="Select cuda device to use. Use -1 for CPU only. (default 0)",
    )

    parser.add_argument('--multiprocess', action='store_true')
    parser.add_argument('--no-multiprocess', dest='multiprocess', action='store_false')
    parser.set_defaults(multiprocess=True)

    return parser.parse_args()


def main() -> None:
    """Predicts OCR for all images with xml annotations in given folder."""
    args = get_args()

    if args.input is None:
        raise ValueError("Please provide an input folder with images and xml files.")

    if args.output is None:
        raise ValueError("Please provide an output folder with prediction xml files.")

    # create output folder if not already existing
    os.makedirs(args.output, exist_ok=True)

    # get file names
    images = list(glob.glob(f'{args.input}/*.jpg'))
    annotations = [f'{args.layout_dir }{os.path.basename(x)[:-4]}.xml' for x in images]

    if args.multiprocess:
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} device(s).")

        model_list = [models.load_any(args.model, device=f"cuda:{i}") for i in range(num_gpus)]

        # Create a pool of worker processes
        with multiprocessing.Pool(processes=num_gpus) as pool:
            tasks = []
            for i, (image_path, annotation_path) in enumerate(zip(images, annotations)):
                output_path = f'{args.output}/{os.path.basename(annotation_path)}'
                if os.path.exists(output_path):
                    continue

                tasks.append((model_list[i % num_gpus],
                              image_path,
                              annotation_path,
                              output_path))

            # Use tqdm to show progress
            for _ in tqdm(pool.starmap(predict, tasks), total=len(tasks)):
                pass

    else:
        device = f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu"
        print(f"Using {device} device.")
        model = models.load_any(args.model, device=device)

        for image_path, annotation_path in tqdm(zip(images, annotations), total=len(images)):
            output_path = f'{args.output}/{os.path.basename(annotation_path)}'
            if os.path.exists(output_path):
                continue

            print(f"OCR for {image_path} using {annotation_path} baselines ...")
            predict(model, image_path, annotation_path, out_path=output_path)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    main()
