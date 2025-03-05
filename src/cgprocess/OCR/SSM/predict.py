"""Module for SSM OCR prediction."""
import argparse
import glob
import json
import lzma
import os
from multiprocessing import Queue, Process
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from bs4 import BeautifulSoup
from ssr import Recognizer, SSMOCRTrainer # pylint: disable=import-error
from torchvision import transforms
from tqdm import tqdm

from src.cgprocess.OCR.SSM.dataset import extract_page
from src.cgprocess.OCR.shared.utils import init_tokenizer, load_cfg, line_has_text
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
        self.manager: Optional[Process] = None

    def join_manager(self) -> None:
        """Join manager process if one is present"""
        if self.manager:
            self.manager.join()
            self.manager = None

    def extract_data(self, image_path: Path, annotations_path: Path, output_path: Path, extension: str) -> Queue:
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
                result_queue.put((file_stem, annotations_path, output_path, False))
                continue
            path_queue.put((file_stem, False))
        total = path_queue.qsize()

        run_processes({"method": get_queue_progress, "args": (total, path_queue)}, processes, path_queue, total,
                      "Preprocessing")
        #
        # self.manager = Process(target=run_processes, args=({"method": get_queue_progress, "args":
        # (total, path_queue)}, processes, path_queue, total, "Preprocessing"))
        # self.manager.start() # needs to be joined externally

        return result_queue


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
        "--num-processes",
        "-p",
        type=int,
        default=None,
        help="Number of processes that are used for preprocessing.",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default="model.pt",
        help="directory with a single .pt or .ckpt file and a single config.yml",
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

    cfg = load_cfg(model_path / "config.yml")

    preprocess = OCRPreprocess(cfg, args.num_processes)

    if not os.path.exists(layout_path / "temp"):
        os.makedirs(layout_path / "temp")
    path_queue = preprocess.extract_data(image_path, layout_path, output_path=layout_path / "temp",
                                         extension=args.extension)

    total = len(image_paths)

    assert len(image_paths) == len(layout_paths), "Images and annotations path numbers do not match."

    num_gpus = torch.cuda.device_count()
    model_list = create_model_list(model_path, num_gpus)

    # files_done = Value('i', 0, lock=True) use this in future to run preprocessing in parallel
    # instead of ahead of prediction

    predictor = MPPredictor("OCR prediction", predict, init_model, path_queue, model_list, str(image_path), True)
    # predictor.launch_processes(num_gpus, total=total, get_progress={"method": get_progress, "args": [files_done, total]})
    predictor.launch_processes(num_gpus, total=total,
                               get_progress={"method": get_queue_progress, "args": (total, path_queue)})

    # preprocess.join_manager()


def predict(args: list, model: Recognizer) -> None:
    """
    Load preprocessed data and run OCR for all text-line crops. Save results inot xml files.
    Args:
        args: list with file stem, annotation path and path for preprocessed data.
        model: State space recognition model
    """
    file_stem, anno_path, data_path, _ = args
    device = model.device

    crops, ids, sorted_indices = load_data(data_path, file_stem)

    batches = len(crops) // model.batch_size
    pred_list: List[str] = []

    for i in range(batches):
        batch = create_batch(crops, sorted_indices, i * model.batch_size, (i + 1) * model.batch_size)
        # for j in range(len(batch)):
        #     image = Image.fromarray(torch.squeeze(batch[j]*255).type(torch.uint8).numpy())
        #     image.save(f"test_output/{j}.png")
        pred_list.extend(model.inference(batch.to(device)))

    if batches * model.batch_size < len(crops):
        diff = model.batch_size - (len(crops) - batches * model.batch_size)
        batch = create_batch(crops, sorted_indices, batches * model.batch_size, None)
        batch = torch.cat([batch, torch.stack([batch[-1].clone()] * diff)])
        pred_list.extend(model.inference(batch.to(device))[:-diff])

    save_results_to_xml(anno_path, file_stem, ids, pred_list)

    # shutil.rmtree(target_path / f"{file_stem}.json", ignore_errors=True)
    # shutil.rmtree(target_path / f"{file_stem}.npz", ignore_errors=True)


def load_data(data_path: Path,
              file_stem: str) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Load preprocessed data and sort crops after their width.
    Return:
        Unsorted crops list, sorted ids and indices for later sorting of crops list.
    """
    with lzma.open(data_path / f"{file_stem}.json", 'r') as file:  # 4. gzip
        json_bytes = file.read()  # 3. bytes (i.e. UTF-8)
    json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
    data = json.loads(json_str)
    crops = []
    length = []
    crops_dict = np.load(data_path / f"{file_stem}.npz")
    for i in range(len(crops_dict)):
        crops.append(crops_dict[str(i)])
        length.append(crops[i].shape[-1])
    sorted_indices = np.argsort(np.array(length))
    ids = np.array(data["ids"])[sorted_indices]
    return crops, ids, sorted_indices


def save_results_to_xml(anno_path: Path, file_stem: str, ids: np.ndarray, pred_list: List[str]) -> None:
    """
    Load xml file and insert ocr results for all test-lines. Text lines are identified via their ids.
    Already present textual data is removed.
    """
    with open(anno_path / f"{file_stem}.xml", "r", encoding="utf-8") as file:
        data = file.read()
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    text_lines = page.find_all('TextLine') # type: ignore
    id_dict = {}
    for i, text_line in enumerate(text_lines):
        id_dict[text_line.attrs["id"]] = i
        if text_line.TextEquiv:
            text_line.TextEquiv.decompose()
    for i, pred_line in enumerate(pred_list):
        bs4_line = text_lines[id_dict[ids[i]]]
        textequiv = soup.new_tag('TextEquiv')
        unicode = soup.new_tag('Unicode')
        unicode.string = pred_line.strip()
        textequiv.append(unicode)
        bs4_line.append(textequiv)
    # save results
    with open(anno_path / f"{file_stem}.xml", 'w', encoding='utf-8') as file:
        file.write(soup.prettify()
                   .replace("<Unicode>\n      ", "<Unicode>")
                   .replace("\n     </Unicode>", "</Unicode>")) # type: ignore


def create_batch(crops: List[np.ndarray], sorted_indices: np.ndarray, start: int, end: Optional[int]) -> torch.Tensor:
    """
    Pad sorted crops to form a batch of crops with uniform width.
    """
    batch = []
    padded_batch = []
    max_width = 0

    for index in sorted_indices[start:end]:
        crop = torch.unsqueeze(torch.tensor(crops[index]), 0).float() / 255
        batch.append(crop)
        width = crop.shape[-1]
        max_width = max(max_width, width)

    for crop in batch:
        if crop.shape[-1] < max_width:
            transform = transforms.Pad((0, 0, max_width - crop.shape[-1], 0))
            padded_batch.append(transform(crop))
        else:
            padded_batch.append(crop)
    return torch.vstack(padded_batch)


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


def create_model_list(model_path: Path, num_gpus: int) -> list:
    """
    Create OCR model list containing one separate model for each process.
    """
    model_list = [[model_path, f"cuda:{i}"] for i in
                  range(num_gpus)] if (
            torch.cuda.is_available() and num_gpus > 0) else \
        [[model_path, "cpu"]]
    return model_list


def init_model(model_path: Path, device: str) -> Recognizer:
    """Init function for compatibility with the MPPredictor handling baseline and layout predictions as well."""
    cfg = load_cfg(model_path / "config.yml")
    tokenizer = init_tokenizer(cfg)
    model = Recognizer(cfg)
    model.tokenizer = tokenizer
    model_path = model_path / [f for f in os.listdir(model_path) if f.endswith(".pt") or f.endswith(".ckpt")][0]
    SSMOCRTrainer.load_from_checkpoint(model_path, model=model,
                                       tokenizer=model.tokenizer,
                                       batch_size=model.batch_size, map_location=device)
    model.device = device
    model.eval()
    return model.to(device)


if __name__ == '__main__':
    main()
