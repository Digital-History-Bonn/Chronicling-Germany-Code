"""Module for SSM OCR prediction."""
import argparse
import glob
import json
import lzma
import os
from multiprocessing import set_start_method, Queue, Process, Value
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from ssr import Recognizer
from tqdm import tqdm

from src.cgprocess.OCR.SSM.dataset import extract_page
from src.cgprocess.shared.multiprocessing_handler import MPPredictor, get_cpu_count, run_processes, get_queue_progress


class OCRPreprocess:
    """Preprocessing includes cropping and masking individual lines for OCR."""

    def __init__(self,
                 cfg: dict,
                 num_processes: Optional[int] = None
                 ):
        """
        Args:
            image_path: path to folder with images
            crop_height: Fixed height. All crops will be scaled to this height
            cfg: configuration file tied to the model
        Attributes:
            manager: Process for running preprocessing asynchronously
        """
        self.image_height = cfg["image_height"]
        self.num_processes = num_processes if num_processes else get_cpu_count() // 8
        self.cfg = cfg
        self.manager = None

    def join_manager(self):
        """Join manager process if one is present"""
        if self.manager:
            self.manager.join()
            self.manager = None

    def extract_data(self, image_path: Path, annotations_path: Path, output_path: Path, extension: str) -> Tuple[
        Queue, int]:
        """Launch processes for preprocessing files, saving them and add them to the path queue."""

        file_stems = [
            f[:-4] for f in os.listdir(annotations_path) if f.endswith(".xml")
        ]

        target_stems = [
            f[:-4] for f in os.listdir(output_path) if f.endswith(".npz")
        ]
        path_queue: Queue = Queue()
        result_queue: Queue = Queue()

        print(f"num processes: {self.num_processes}")
        processes = [Process(target=extract_page,
                             args=(path_queue, (image_path, annotations_path, output_path),
                                   extension,
                                   self.cfg, result_queue, True)) for _ in range(self.num_processes)]

        for file_stem in tqdm(file_stems, desc="Put paths in queue"):
            if file_stem in target_stems:
                continue
            path_queue.put((file_stem, False))
        total = path_queue.qsize()

        self.manager = Process(target=run_processes, args=({"method": get_queue_progress, "args": (total, path_queue)}, processes, path_queue, total, "Preprocessing"))
        self.manager.start() # needs to be joined externally

        return result_queue, total


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Predict SSM OCR")
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="default-ocr-prediction",
        help="Name of the model and the log files."
    )
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        help="path for folder with folders 'images' and 'annotations'",
    )
    parser.add_argument(
        "--layout-path",
        "-l",
        type=str,
        help="path for folder with layout xml files."
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=None,
        help="Number of processes that are used for preprocessing.",
    )
    # parser.add_argument(
    #     "--thread-count",
    #     "-t",
    #     type=int,
    #     default=1,
    #     help="Select number of threads that are launched per process.",
    # )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default="model.pt",
        help="path to model .pt or .ckpt file",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".jpg",
        help="image extension default '.jpg'",
    )

    return parser.parse_args()


def main() -> None:
    """Predicts OCR for all images with xml annotations in given folder."""
    args = get_args()
    image_path = Path(args.image_path)
    layout_path = Path(args.layout_path)
    model_path = Path(args.model_path)

    # get file names
    image_paths = list(glob.glob(f'{image_path}/*.jpg'))
    layout_paths = [f'{layout_path}/{os.path.basename(path)[:-4]}.xml' for path in image_paths]

    cfg = torch.load(model_path).cfg #todo: is this valid?

    preprocess = OCRPreprocess(cfg, args.num_processes)
    path_queue, total = preprocess.extract_data(image_path, layout_path, output_path=layout_path / "temp",
                                                extension=args.extension)

    assert len(image_paths) == len(layout_paths), "Images and annotations path numbers do not match."

    num_gpus = torch.cuda.device_count()

    model_list = create_model_list(model_path, cfg, num_gpus)
    files_done = Value('i', 0, lock=True)

    predictor = MPPredictor("OCR prediction", predict, init_model, path_queue, model_list, str(image_path), True)
    predictor.launch_processes(num_gpus, total=total, get_progress={"method": get_progress, "args": [files_done, total]})

    preprocess.join_manager()

def predict(args, model) -> None:
    file_stem, anno_path, target_path, _ = args # todo rename target path everywhere for prediction. THose are preprocessed data and no ML targets.

    with lzma.open(target_path / f"{file_stem}.json", 'r') as file:  # 4. gzip
        json_bytes = file.read()  # 3. bytes (i.e. UTF-8)

    json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
    data = json.loads(json_str)

    crops = []
    length = []
    crops_dict = np.load(target_path / f"{file_stem}.npz")
    for i in range(len(crops_dict)):
        crops.append(crops_dict[str(i)])
        length.append(len(crops[i]))

    ids = data["ids"]

    #todo: sort crops

    pred = model.inference(crops, batch_size, tokenizer)

    # todo: save xml


def get_progress(files_done: Synchronized, total) -> int:
    """Returns value of shared variable, or the supplied total to indicade processing is done, if value is < 0."""
    value = files_done.value
    return total if value < 0 else value


def create_path_queue(annotations: List[str], images: List[str]) -> Queue:
    """
    Create path queue for OCR prediction containing image and layout path.
    Elements are required to have the image path at
    index 0 and the bool variable for terminating processes at index -1.
    :param annotations: list of annotation paths
    :param images: list of image paths
    """
    path_queue: Queue = Queue()
    for image_path, annotation_path in zip(images, annotations):
        path_queue.put((image_path,
                        annotation_path,
                        False))
    return path_queue


def create_model_list(model_path: Path, num_gpus: int, cfg) -> list:
    """
    Create OCR model list containing one separate model for each process.
    """
    model_list = [[model_path, cfg, f"cuda:{i}"] for i in
                   range(num_gpus)] if (
            torch.cuda.is_available() and num_gpus > 0) else \
        [[model_path, cfg, "cpu"]]
    return model_list #todo: do this without cfg, if it is loadable


def init_model(model_path: Path, cfg: dict, device: str) -> Recognizer:
    """Init function for compatibility with the MPPredictor handling baseline and layout predictions as well."""
    model = Recognizer(cfg) #todo: do this without cfg, if it is loadable
    return model.load(model_path, device=device)


if __name__ == '__main__':
    set_start_method('spawn')
    main()
