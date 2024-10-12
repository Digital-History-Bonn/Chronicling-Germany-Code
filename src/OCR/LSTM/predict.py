"""OCR prediction module."""

import argparse
import glob
import os
from multiprocessing import Process, Queue, set_start_method
from threading import Thread
from time import sleep
from typing import Tuple, List

import torch
from PIL import Image
from bs4 import BeautifulSoup, PageElement
from kraken import rpred  # pylint: disable=import-error
from kraken.containers import Segmentation, \
    BaselineLine  # pylint: disable=no-name-in-module, import-error
from kraken.lib import models  # pylint: disable=import-error
from kraken.lib.models import TorchSeqRecognizer  # pylint: disable=import-error
from tqdm import tqdm

from src.OCR.utils import pad_image, adjust_path, pad_points


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
            # remove text, to make place for ocr prediction
            if text_line.TextEquiv:
                text_line.TextEquiv.decompose()

            polygon_points = pad_points(text_line.find('Coords')['points'])
            baseline = text_line.find('Baseline')
            if baseline:
                baseline_points = pad_points(baseline['points'])
                # Split the points string and convert them to a list of tuples
                region_baselines.append(
                    BaselineLine(
                        str(i),
                        [tuple(map(int, point.split(','))) for point in
                         baseline_points.split()],  # type: ignore
                        [tuple(map(int, point.split(','))) for point in
                         polygon_points.split()])  # type: ignore
                )

        baselines.append(region_baselines)

    return soup, text_regions, baselines


def predict_batch(model: TorchSeqRecognizer, path_queue: Queue, thread_count: int = 1) -> None:
    """
    Predicts OCR on given image using given baseline annotations and model.
    Takes paths from a multiprocessing queue and must be terminated externally when all paths have been processed.
    Args:
        model: Model to use for OCR prediction.
        path_queue: multiprocessing queue for path tuples.
        thread_count: this number of threads will run predictions in parallel. This might lead to an CUDA out of
        memory error if too many threads are launched.
    """
    while True:
        threads: List[Thread] = []
        for i in range(thread_count):
            image_path, anno_path, out_path, done = path_queue.get()
            if done:
                join_threads(threads)
                return

            threads.append(Thread(target=predict, args=(anno_path, image_path, model, out_path)))
            threads[i].start()
        for thread in threads:
            thread.join()


def join_threads(threads: List[Thread]) -> None:
    """
    Join all threads.
    """
    for thread in threads:
        thread.join()


def predict(anno_path: str, image_path: str, model: TorchSeqRecognizer, out_path: str) -> None:
    """
    Predicts OCR on given image using given baseline annotations and model.
    Takes paths from a multiprocessing queue and must be terminated externally when all paths have been processed.
    """
    print(f'Predicting {image_path}...')
    # load image and pad image
    im = pad_image(Image.open(image_path))
    # preprocess annotations and extract baselines
    file_name = os.path.basename(image_path)
    soup, regions, region_baselines = extract_baselines(anno_path)
    for region, baselines in zip(regions, region_baselines):
        baseline_seg = Segmentation(type='baselines',
                                    imagename=file_name,
                                    text_direction='horizontal-lr',
                                    script_detection=False,
                                    lines=baselines,
                                    line_orders=[])

        # single model recognition
        pred_it = rpred.rpred(model, im, baseline_seg, no_legacy_polygons=True)
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
        help="path for folder with images. Need to be jpg."
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

    parser.add_argument(
        "--thread-count",
        "-t",
        type=int,
        default=1,
        help="Select number of threads that are launched per process. This must be used carefully, as it can "
             "lead to a CUDA out of memory error.",
    )

    parser.add_argument(
        "--process-count",
        "-p",
        type=int,
        default=1,
        help="Select number of processes that are launched per graphics card. This must be used carefully, as it can "
             "lead to a CUDA out of memory error.",
    )

    return parser.parse_args()


def main() -> None:
    """Predicts OCR for all images with xml annotations in given folder."""
    args = get_args()
    input_path = adjust_path(args.input)
    layout_path = adjust_path(args.layout_dir)
    output_path = adjust_path(args.output)

    if input_path is None:
        raise ValueError("Please provide an input folder with images and xml files.")

    if output_path is None:
        raise ValueError("Please provide an output folder.")

    # create output folder if not already existing
    os.makedirs(output_path, exist_ok=True)

    # get file names
    images = list(glob.glob(f'{input_path}/*.jpg'))
    annotations = [f'{layout_path}/{os.path.basename(x)[:-4]}.xml' for x in images]

    assert len(images) == len(annotations), "Images and annotations path numbers do not match."

    num_gpus = torch.cuda.device_count()
    # pylint: disable=duplicate-code
    if num_gpus > 0:
        print(f"Using {num_gpus} gpu device(s).")
    else:
        print("Using cpu.")

    path_queue: Queue = Queue()

    # put paths in queue
    for image_path, annotation_path in zip(images, annotations):
        output_path = f'{args.output}/{os.path.basename(annotation_path)}'
        path_queue.put((image_path,
                        annotation_path,
                        output_path,
                        False))

    model_list = [models.load_any(args.model, device=f"cuda:{i % num_gpus}") for i in
                  range(num_gpus * args.process_count)] if (
                torch.cuda.is_available() and num_gpus > 0) else \
        [models.load_any(args.model, device="cpu")]

    processes = [Process(target=predict_batch,
                         args=(model_list[i if num_gpus > 0 else 0], path_queue, args.thread_count))
                         for i in range(len(model_list))]
    for process in processes:
        process.start()
    total = len(images)
    # pylint: disable=duplicate-code
    with tqdm(total=path_queue.qsize(), desc="OCR prediction", unit="pages") as pbar:
        while not path_queue.empty():
            pbar.n = total - path_queue.qsize()
            pbar.refresh()
            sleep(1)
    for _ in processes:
        path_queue.put(("", "", "", True))
    for process in tqdm(processes, desc="Waiting for processes to end"):
        process.join()


if __name__ == '__main__':
    set_start_method('spawn')
    main()
