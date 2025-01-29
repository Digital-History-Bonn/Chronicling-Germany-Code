"""Script for extracting target and input data for OCR SSM training. This is done with multiprocessing to acelerate
the xml reading."""
import argparse
import os
from multiprocessing import Queue, Process
from pathlib import Path

from tqdm import tqdm

from src.cgprocess.OCR.SSM.dataset import load_data, preprocess_data
from src.cgprocess.OCR.shared.tokenizer import OCRTokenizer
from src.cgprocess.shared.multiprocessing_handler import run_processes
from src.cgprocess.shared.utils import prepare_file_loading


def main(args: argparse.Namespace) -> None:
    """Load xml files and save result image.
    Calls read and draw functions"""
    annotations_path = Path(args.data_path) / "annotations"
    image_path = Path(args.data_path) / "images"
    output_path = Path(args.data_path) / "targets"

    file_stems = [
        f[:-4] for f in os.listdir(annotations_path) if f.endswith(".xml")
    ]

    if not output_path or not os.path.exists(output_path):
        print(f"creating {output_path}.")
        os.makedirs(output_path)  # type: ignore

    target_stems = [
        f[:-4] for f in os.listdir(output_path) if f.endswith(".npz")
    ]
    path_queue: Queue = Queue()
    total = len(file_stems)

    extension, _ = prepare_file_loading(args.dataset)
    tokenizer = OCRTokenizer()

    processes = [Process(target=extract_data, args=(path_queue, target_stems, image_path, annotations_path, extension, tokenizer)) for _ in range(32)]

    for path in tqdm(file_stems, desc="Put paths in queue"):
        path_queue.put((path, False))

    run_processes({"method": get_progress, "args": output_path}, processes, path_queue, total, "Page converting")


def extract_data(queue: Queue, target_paths: list, image_path: Path, target_path: Path, image_extension: str,
                 tokenizer: OCRTokenizer, image_height) -> None:
    """Extract target and input data for OCR SSM training"""
    while True:
        arguments = queue.get()
        if arguments[-1]:
            break
        file_stem, _ = arguments
        if file_stem in target_paths:
            return
        image, text_lines = load_data(image_path / f"{file_stem}{image_extension}",
                                      target_path / f"{file_stem}.xml")
        crops, texts = preprocess_data(image, text_lines, image_height)
        targets = [tokenizer(line) for line in texts]


def get_progress(output_path):
    len([f for f in os.listdir(output_path) if f.endswith(".npz")])


def get_args() -> argparse.Namespace:
    """defines arguments"""
    # pylint: disable=locally-disabled, duplicate-code
    parser = argparse.ArgumentParser(description="creates targets from annotation xmls")
    parser.add_argument(
        "--dataset",
        type=str,
        default="transkribus",
        help="select dataset to load " "(transkribus, HLNA2013)",
    )
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        help="path for folder with 'images' and 'annotations' folders. ",
    )

    return parser.parse_args()

 # todo: this should not be a scipt but called from training code. If files already exist and are in line with model
 #  config, dont reconvert.
if __name__ == "__main__":
    args = get_args()
    main(args)
