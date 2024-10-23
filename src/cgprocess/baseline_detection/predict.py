"""Prediction script for Pero baseline detection."""
import argparse
import glob
import os
import random
from multiprocessing import Queue, Process, set_start_method
from threading import Thread
from time import sleep
from typing import List, Tuple

import numpy as np
import torch
from shapely.geometry import Polygon, LineString
from torch import nn, Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from scipy import ndimage
from PIL import ImageDraw, Image
from bs4 import BeautifulSoup

from skimage import draw

from monai.networks.nets import BasicUNet
from tqdm import tqdm

from src.cgprocess.OCR.LSTM.predict import join_threads
from src.cgprocess.baseline_detection.utils import nonmaxima_suppression, adjust_path, add_baselines, load_image


def baseline_to_textline(baseline: np.ndarray, heights: List[float]) -> np.ndarray:
    """
    From https://github.com/DCGM/pero-ocr/blob/master/pero_ocr/layout_engines/layout_helpers.py.

    Convert baseline coords and its respective heights to a textline polygon.

    Args:
        baseline: baseline coords
        heights: textline heights

    Returns:
        textline polygon
    """
    heights = np.array([max(1, heights[0]), max(1, heights[1])]).astype(np.float32)  # type: ignore

    x_diffs = np.diff(baseline[:, 0])
    x_diffs = np.concatenate((x_diffs, x_diffs[-1:]), axis=0)
    y_diffs = np.diff(baseline[:, 1])
    y_diffs = np.concatenate((y_diffs, y_diffs[-1:]), axis=0)

    alfas = np.pi / 2 + np.arctan2(y_diffs, x_diffs)
    y_up_diffs = np.sin(alfas) * heights[0]
    x_up_diffs = np.cos(alfas) * heights[0]
    y_down_diffs = np.sin(alfas) * heights[1]
    x_down_diffs = np.cos(alfas) * heights[1]

    pos_up = baseline.copy().astype(np.float32)
    pos_up[:, 1] -= y_up_diffs
    pos_up[:, 0] -= x_up_diffs
    pos_down = baseline.copy().astype(np.float32)
    pos_down[:, 1] += y_down_diffs
    pos_down[:, 0] += x_down_diffs
    pos_t = np.concatenate([pos_up, pos_down[::-1, :]], axis=0)

    return pos_t  # type: ignore


def order_lines_vertical(baselines: List[np.ndarray],
                         heights: List[List[float]],
                         textlines: List[np.ndarray]) -> Tuple[List[np.ndarray],
List[List[float]],
List[np.ndarray]]:
    """
    From https://github.com/DCGM/pero-ocr/blob/master/pero_ocr/layout_engines/layout_helpers.py.

    Order lines according to their vertical position.

    Args:
        baselines: list of baselines to order
        heights: list of respective textline heights
        textlines: list of respective textline polygons

    Returns:
        ordered baselines, heights and textlines
    """
    # adding random number to order to prevent swapping when two lines are on same y-coord
    baselines_order = [baseline[0][1] + random.uniform(0.001, 0.999) for baseline in baselines]
    baselines = [baseline for _, baseline in sorted(zip(baselines_order, baselines))]
    heights = [height for _, height in sorted(zip(baselines_order, heights))]
    textlines = [textline for _, textline in sorted(zip(baselines_order, textlines))]

    return baselines, heights, textlines


def extract_layout(xml_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extracts the textregions and regions to mask form xml file.

    Args:
        xml_path: path to xml file

    Returns:
        mask regions and textregions
    """
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    mask_regions = []
    rois = []

    table_regions = page.find_all(['TableRegion'])
    for region in table_regions:
        coords = region.find('Coords')
        points = np.array([tuple(map(int, point.split(','))) for
                               point in coords['points'].split()])
        mask_regions.append(points)

    text_regions = page.find_all(['TextRegion'])
    for region in text_regions:
        coords = region.find('Coords')
        points = np.array([tuple(map(int, point.split(','))) for
                               point in coords['points'].split()])
        rois.append(points)

    return mask_regions, rois


def preprocess_image(image: torch.Tensor,
                     mask_regions: List[np.ndarray]) -> torch.Tensor:
    """
    Preprocesses the image for prediction.

    Args:
        image: input image as torch tensor
        mask_regions: List of torch tensors, each representing region to mask

    Returns:
        preprocessed image
    """
    _, width, height = image.shape
    mask = torch.ones(width, height)  # mask to filter regions

    # draw mask
    for mask_region in mask_regions:
        # draw mask to remove not text regions
        if len(mask_region) >= 3:
            rr, cc = draw.polygon(mask_region[:, 1], mask_region[:, 0], shape=(width, height))
            mask[rr, cc] = 0

    # preprocess image
    image *= mask
    resize = transforms.Resize((width // 2, height // 2))
    return resize(image)  # type: ignore


def apply_polygon_mask(image: torch.Tensor, roi: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Masks everything outside the roi polygone with zeros.

    Args:
        image: torch Tensor of the shape (channel, width, height)
        roi: torch Tensor with the polygon points

    Returns:
        image: masked Tensor
    """

    bounds = list(Polygon(roi).bounds)
    if bounds[0] < 0:
        bounds[0] = 0
    if bounds[1] < 0:
        bounds[1] = 0
    if bounds[2] >= image.shape[2]:
        bounds[2] = image.shape[2]-1
    if bounds[3] >= image.shape[1]:
        bounds[3] = image.shape[1]-1
    offset = np.array([bounds[0], bounds[1]], dtype=int)
    shape = int(bounds[3] - bounds[1]), int(bounds[2] - bounds[0])
    mask = draw.polygon2mask(shape, roi[:, ::-1]-offset[::-1])

    result = image[:, int(bounds[1]): int(bounds[3]), int(bounds[0]): int(bounds[2])]
    result[:, mask == 0] = 0.0
    return result, offset


class BaselineEngine:
    """Class to predict baselines using approach from: https://arxiv.org/abs/2102.11838."""

    def __init__(self, model_name: str,
                 cuda: int = 0,
                 downsample: float = 1.0,
                 smooth_line_predictions: bool = True,
                 line_end_weight: float = 1.0,
                 line_detection_threshold: float = 0.2,
                 vertical_line_connection_range: int = 5,
                 paragraph_line_threshold: float = 0.3,
                 thread_count: int = 1
                 ):
        """
        Predicts baselines using approach from: https://arxiv.org/abs/2102.11838.

        Args:
            model_name: model name to load
            cuda: CUDA device
            downsample: downsample factor for images before prediction (default: 1.0)
            smooth_line_predictions: Smooth line predictions (default: True)
            line_end_weight: Line end weight (default: 1.0)
            line_detection_threshold: Line detection threshold (default: 0.2)
            vertical_line_connection_range: Vertical line connection range (default: 5)
            paragraph_line_threshold: Paragraph line threshold (default: 0.3)
        """
        self.thread_count = thread_count
        self.downsample = downsample
        self.smooth_line_predictions = smooth_line_predictions
        self.line_end_weight = line_end_weight
        self.line_detection_threshold = line_detection_threshold
        self.vertical_line_connection_range = vertical_line_connection_range
        self.paragraph_line_threshold = paragraph_line_threshold

        self.device = (
            torch.device(f"cuda:{cuda}")
            if torch.cuda.is_available() and cuda >= 0
            else torch.device("cpu")
        )

        self.model = BasicUNet(spatial_dims=2, in_channels=3, out_channels=6).to(self.device)
        self.model.load_state_dict(
            torch.load(f"{model_name}.pt",
                       map_location=self.device
                       )
        )
        self.model.eval()

        # Define transformations
        self.to_tensor = transforms.ToTensor()
        self.softmax = nn.Softmax(dim=1)

    def draw_textregions(self, textregions: List[torch.Tensor],
                         width: int,
                         height: int) -> torch.Tensor:
        """
        Creates outline image of the textregions from our layout prediction.

        This is similar to the Text outline output of the Transformer model.

        Args:
            textregions: List of textregions as polygons to draw in torch tensor
            width: width of image
            height: height of image

        Returns:
            textregions outlines drawn in torch tensor
        """
        # draw textregions
        textregion_img = Image.new('L', (height, width), color=0)
        textregion_draw = ImageDraw.Draw(textregion_img)
        for textregion in textregions:
            # draw textregion
            textregion_draw.polygon([(x[1].item(), x[0].item()) for x in textregion],
                                    fill=0,
                                    outline=255,
                                    width=3)

        return self.to_tensor(textregion_img)  # type: ignore

    def parse(self, out_map: np.ndarray) -> Tuple[List[np.ndarray],
    List[List[float]],
    List[np.ndarray]]:
        """
        From
        https://github.com/DCGM/pero-ocr/blob/master/pero_ocr/layout_engines/cnn_layout_engine.py.

        Parse input baseline, height and region map into list of baselines
        coords, list of heights and region map

        Args:
            out_map: array of baseline and endpoint probabilities with

        Returns:
            List of baselines,
            List of baseline heights and depth,
            List of textline polygons
        """
        b_list = []
        h_list = []

        # expand line heights verticaly
        heights_map = ndimage.grey_dilation(
            out_map[:, :, :2], size=(5, 1, 1))

        baselines_map = out_map[:, :, 2]
        if self.smooth_line_predictions:
            baselines_map = ndimage.convolve(baselines_map, np.ones((3, 3)) / 9)
        baselines_map = nonmaxima_suppression(baselines_map, element_size=(5, 1))
        baselines_map = baselines_map - self.line_end_weight * out_map[:, :, 3]
        baselines_map = baselines_map > self.line_detection_threshold

        # connect vertically disconnected lines - any effect?
        # Parameter is vertical connection distance in pixels.
        baselines_map_dilated = ndimage.binary_dilation(
            baselines_map,
            structure=np.asarray([[1, 1, 1] for _ in range(self.vertical_line_connection_range)]))
        baselines_img, num_detections = ndimage.label(baselines_map_dilated,
                                                      structure=np.ones([3, 3]))
        baselines_img *= baselines_map
        inds = np.where(baselines_img > 0)
        labels = baselines_img[inds[0], inds[1]]

        for i in range(1, num_detections + 1):
            bl_inds, = np.where(labels == i)
            if len(bl_inds) > 5:
                # go from matrix indexing to image indexing
                pos_all = np.stack([inds[1][bl_inds], inds[0][bl_inds]], axis=1)

                _, indices = np.unique(pos_all[:, 0], return_index=True)
                pos = pos_all[indices]
                x_index = np.argsort(pos[:, 0])
                pos = pos[x_index]

                target_point_count = min(10, pos.shape[0] // 10)
                target_point_count = max(target_point_count, 2)
                selected_pos = np.linspace(
                    0, (pos.shape[0]) - 1, target_point_count).astype(np.int32)

                pos = pos[selected_pos, :]
                pos[0, 0] -= 2  # compensate for endpoint detection overlaps
                pos[-1, 0] += 2

                heights_pred = heights_map[inds[0][bl_inds], inds[1][bl_inds], :]
                heights_pred = np.maximum(heights_pred, 0)
                heights_pred = np.asarray([
                    np.percentile(heights_pred[:, 0], 50),
                    np.percentile(heights_pred[:, 1], 50)
                ])

                b_list.append(self.downsample * pos.astype(float))
                h_list.append([self.downsample * heights_pred[0],
                               self.downsample * heights_pred[1]])

        # sort lines from LEFT to RIGHT
        x_inds = [np.amin(baseline[:, 0]) + 0.0001 * np.random.rand() for baseline in b_list]
        b_list = [b for _, b in sorted(zip(x_inds, b_list))]
        h_list = [h for _, h in sorted(zip(x_inds, h_list))]

        t_list = [baseline_to_textline(b, h) for b, h in zip(b_list, h_list)]

        return b_list, h_list, t_list

    def predict(self, image: torch.Tensor, layout: str) -> Tuple[List[List[Polygon]], List[List[LineString]]]:
        """
        Predicts the baselines and textlines on a given image.

        Args:
            image: torch Tensor of image (channel, width, height)
            layout: path to layout xml file

        Returns:
            predicted textlines and baselines
        """
        _, width, height = image.shape
        baseline_lst: List[List[LineString]] = []
        textline_lst: List[List[Polygon]] = []

        # extract layout information
        mask_regions, text_regions = extract_layout(layout)
        input_image = preprocess_image(image, mask_regions)

        # predict
        input_image = input_image[None].to(self.device)
        pred_gpu = self.model(input_image)  # pylint: disable=not-callable
        pred = pred_gpu.cpu().detach()
        del pred_gpu, input_image

        # extract maps
        ascenders = pred[0, 0]
        descenders = pred[0, 1]
        baselines = self.softmax(pred[:, 2:4])[0, 1]
        limits = self.softmax(pred[:, 4:6])[0, 1]
        maps = torch.stack([ascenders, descenders, baselines, limits])

        # resize to image size
        resize = transforms.Resize((width, height), interpolation=InterpolationMode.NEAREST)
        maps = resize(maps)

        threads: List[Thread] = []
        for roi in text_regions:
            if len(roi) < 3:
                continue
            if len(threads) >= self.thread_count:
                join_threads(threads)
                threads = []

            threads.append(Thread(target=self.postprocess_per_region, args=(baseline_lst, maps, roi, textline_lst)))
            threads[-1].start()
        join_threads(threads)

        return textline_lst, baseline_lst

    def postprocess_per_region(self, baseline_lst: List[List[LineString]], prediction: Tensor, roi: np.ndarray,
                               textline_lst: List[List[Polygon]]) -> None:
        """
        Args:
            prediction: prediction Tensor with channels for ascenders, descenders, baselines and limits.
            roi: Tensor containing polygon points.
        Postprocessing is applied for each region separately, to ensure all lines are within one region.
        Therefore, all but the region polygon is masked within the prediction.
        """
        mask_map, offset = apply_polygon_mask(prediction, roi)
        # postprocess from Transformer
        b_list, h_list, t_list = self.parse(mask_map.permute(1, 2, 0).numpy())
        # b_list, h_list, t_list = order_lines_vertical(b_list, h_list, t_list)
        baseline_lst.append([LineString(line + offset).simplify(tolerance=1) for line in b_list])
        textline_lst.append([Polygon(poly + offset).simplify(tolerance=1) for poly in t_list])


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="predict")
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default=None,
        help="path for folder with images. Images need to be jpg."
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
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="path to the folder where to save the preprocessed files",
    )

    # pylint: disable=duplicate-code
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="path to the model file",
    )

    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=1,
        help="Number of processes for every gpu in use. This has to be used carefull, as this can lead to a CUDA out "
             "of memory error.",
    )

    parser.add_argument(
        "--thread-count",
        "-t",
        type=int,
        default=1,
        help="Select number of threads that are launched per process.",
    )

    return parser.parse_args()


def main() -> None:
    """Predicts textlines and baselines for all files in given folder."""
    args = get_args()

    input_dir = adjust_path(args.input_dir)
    layout_dir = adjust_path(args.layout_dir)
    output_dir = adjust_path(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = list(glob.glob(f"{input_dir}/*.jpg"))
    layout_xml_paths = [f"{layout_dir}/{os.path.basename(i)[:-4]}.xml" for i in image_paths]
    output_files = [f"{output_dir}/{os.path.basename(i)[:-4]}.xml" for i in image_paths]

    num_gpus = torch.cuda.device_count()
    num_processes = args.processes
    if num_gpus > 0:
        print(f"Using {num_gpus} gpu device(s).")
    else:
        print("Using cpu.")

    # create queue
    path_queue: Queue = Queue()
    for image, layout, output_file in zip(image_paths, layout_xml_paths, output_files):
        path_queue.put((image, layout, output_file, False))

    device_ids = list(range(num_gpus) if torch.cuda.is_available() else [-1])
    models = [BaselineEngine(model_name=args.model, cuda=device_ids[i % num_gpus], thread_count=args.thread_count)
              for i in range(len(device_ids) * num_processes)]
    processes = [Process(target=predict, args=(models[i], path_queue)) for i in range(len(models))]
    for process in processes:
        process.start()
    total = len(image_paths)
    # pylint: disable=duplicate-code
    with tqdm(total=path_queue.qsize(), desc="Baseline Prediction", unit="pages") as pbar:
        while not path_queue.empty():
            pbar.n = total - path_queue.qsize()
            pbar.refresh()
            sleep(1)
    for _ in processes:
        path_queue.put(("", "", "", True))
    for process in tqdm(processes, desc="Waiting for processes to end"):
        process.join()


def predict(model: BaselineEngine, path_queue: Queue) -> None:
    """
    Predicts baselines for given image with given layout and writes into outputfile.
    Takes paths from a multiprocessing queue and must be terminated externally when all paths have been processed.
    Args:
        path_queue (Queue): Queue object containing image, layout and output path and info if job is already done
        model (str): Path to model file
    """
    while True:
        image_path, layout_xml_path, output_file, done = path_queue.get()
        if done:
            break
        image = load_image(image_path)
        textlines, baselines = model.predict(image, layout_xml_path)
        add_baselines(
            layout_xml=layout_xml_path,
            textlines=textlines,
            baselines=baselines,
            output_file=output_file
        )


if __name__ == '__main__':
    set_start_method('spawn')
    main()
