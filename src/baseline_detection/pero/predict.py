"""Prediction script for Pero baseline detection."""

from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import torch
from shapely import Polygon, LineString
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import ImageDraw, Image
from bs4 import BeautifulSoup

from skimage import draw, io

from monai.networks.nets import BasicUNet

from src.baseline_detection.utils import get_tag, nonmaxima_suppression
from src.baseline_detection.xml_conversion import add_baselines


def baseline_to_textline(baseline: np.ndarray, heights: List[float, float]) -> np.ndarray:
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

    return pos_t


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


def plot_lines_on_image(image: torch.Tensor,
                        baselines: List[torch.Tensor],
                        textlines: List[torch.Tensor]) -> None:
    """
    Plot lines on the given image.

    Args:
        image: Torch tensor representing the image.
        baselines: List of torch tensors, each containing the coordinates of one baseline
        textlines: List of torch tensors, each containing the coordinates of one textlines polygon

    """
    # Create a copy of the image to avoid modifying the original
    _, w, h = image.shape
    baseline_image = np.zeros((w, h))
    textline_image = np.zeros((w, h))

    # Iterate through the list of lines and draw each line on the image
    for line in baselines:
        for i in range(len(line) - 1):
            x1, y1 = line[i]
            x2, y2 = line[i + 1]
            rr, cc = draw.line(int(y1), int(x1), int(y2), int(x2))
            baseline_image[rr, cc] = 1  # RGB color for the lines (red)

    for polygon_coords in textlines:
        # Extract x and y coordinates of the polygon vertices
        rr, cc = draw.polygon_perimeter(polygon_coords[:, 1], polygon_coords[:, 0], shape=(w, h))
        textline_image[rr, cc] = 1

    # Plot the image with lines
    plt.imshow(image.permute(1, 2, 0))
    plt.imshow(textline_image, cmap='Reds', alpha=0.4)
    plt.imshow(baseline_image, cmap='Blues', alpha=0.4)

    for i, line in enumerate(baselines):
        plt.text(line[0][0].item(), line[0][1].item(), str(i), fontsize=1, color='red')

    plt.axis('off')  # Turn off axis
    plt.tight_layout()
    plt.savefig(f"{Path(__file__).parent.absolute()}/../../../"
                f"data/PeroBaselinePrediction7.png", dpi=1000)
    print(f"saved fig to {Path(__file__).parent.absolute()}/../../../"
          f"data/PeroBaselinePrediction7.png")


def get_textregions(xml_path: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Extracts the textregions and regions to mask form xml file.

    Some regions like table and Header look like text, but we don't want to predict baselines
    here, so we mask them for prediction.

    Args:
        xml_path: path to xml file

    Returns:
        textregions and mask regions
    """
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    mask_regions = []
    textregions = []

    text_regions = page.find_all('TextRegion')
    for region in text_regions:
        tag = get_tag(region)

        if tag in ['table', 'header']:
            coords = region.find('Coords')
            points = torch.tensor([tuple(map(int, point.split(','))) for
                                   point in coords['points'].split()])
            mask_regions.append(points)

        if tag in ['heading', 'article_', 'caption', 'paragraph']:
            coords = region.find('Coords')
            textregion = torch.tensor([tuple(map(int, point.split(','))) for
                                       point in coords['points'].split()])
            textregion = textregion[:, torch.tensor([1, 0])]
            textregions.append(textregion)

    return textregions, mask_regions


class BaselineEngine:
    """Class to predict baselines using approach from: https://arxiv.org/abs/2102.11838."""

    def __init__(self, model_name: str,
                 cuda: int = 0,
                 downsample: float = 1.0,
                 smooth_line_predictions: bool = True,
                 line_end_weight: float = 1.0,
                 line_detection_threshold: float = 0.2,
                 vertical_line_connection_range: int = 5,
                 paragraph_line_threshold: float = 0.3
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
        self.downsample = downsample
        self.smooth_line_predictions = smooth_line_predictions
        self.line_end_weight = line_end_weight,
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
            torch.load(
                f"{Path(__file__).parent.absolute()}/../../../models/{model_name}.pt",
                map_location=self.device
            )
        )

        # Define transformations
        self.to_tensor = transforms.ToTensor()
        self.softmax = nn.Softmax(dim=1)

    def preprocess_image(self, image: torch.Tensor,
                         mask_regions: List[torch.Tensor]) -> torch.Tensor:
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
        return resize(image)

    def draw_textregions(self, textregions: List[torch.Tensor],
                         width: int,
                         height: int) -> torch.Tensor:
        """
        Creates outline image of the textregions from our layout prediction.

        This is similar to the Text outline output of the pero model.

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

        return self.to_tensor(textregion_img)

    def parse(self, out_map: np.ndarray) -> Tuple[List[np.ndarray],
                                                  List[List[float]],
                                                  List[np.ndarray]]:
        """
        From https://github.com/DCGM/pero-ocr/blob/master/pero_ocr/layout_engines/cnn_layout_engine.py.

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

        print('MAP RES:', out_map.shape)
        out_map[:, :, 4][out_map[:, :, 4] < 0] = 0

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

    def predict(self, image: torch.Tensor, layout: str) -> Tuple[List[Polygon], List[LineString]]:
        """
        Predicts the baselines and textlines on a given image.

        Args:
            image: torch Tensor of image (channel, width, height)
            layout: path to layout xml file

        Returns:
            predicted textlines and baselines
        """
        _, width, height = image.shape
        maps = torch.zeros((5, width, height))  # tensor to save predictions from networks

        # extract layout information
        textregions, mask_regions = get_textregions(layout)
        maps[4] = self.draw_textregions(textregions, width, height)
        input_image = self.preprocess_image(image, mask_regions)

        # predict
        pred = self.model(input_image[None].to(self.device))
        pred = pred.cpu().detach()
        ascenders = pred[0, 0]
        descenders = pred[0, 1]
        baselines = self.softmax(pred[:, 2:4])[0, 1]
        limits = self.softmax(pred[:, 4:6])[0, 1]

        pred = torch.stack([ascenders, descenders, baselines, limits])
        resize = transforms.Resize((width, height), interpolation=InterpolationMode.NEAREST)
        maps[:4] = resize(pred)

        # postprocess from pero
        b_list, h_list, t_list = self.parse(maps.permute(1, 2, 0).numpy())

        if not b_list:
            print('fail!')
            return [], []

        # clusters_array = make_clusters(b_list, h_list, t_list, maps[:, :, 4], 2)
        # p_list = clustered_lines_to_polygons(t_list, clusters_array)

        b_list, h_list, t_list = order_lines_vertical(b_list, h_list, t_list)
        # p_list, b_list, t_list = rotate_layout(p_list, b_list, t_list, rot, image.shape)

        baselines = [LineString(line[:, ::-1]).simplify(tolerance=1) for line in b_list]
        textlines = [Polygon(poly[:, ::-1]).simplify(tolerance=1) for poly in t_list]

        return textlines, baselines


def main(image_path: str, layout_xml_path: str, output_file: str) -> None:
    """
    Predicts textlines and baselines in given image and writes into an annotation xml file.

    Args:
        image_path (str): Path to image
        layout_xml_path (str): Path to layout xml file
        output_file (str): Path to output xml file
    """
    baseline_engine = BaselineEngine(model_name='height2_baseline_e250_es', cuda=0)

    image = torch.tensor(io.imread(image_path)).permute(2, 0, 1) / 256
    textlines, baselines = baseline_engine.predict(image, layout_xml_path)

    add_baselines(
        layout_xml=layout_xml_path,
        textlines=textlines,
        baselines=baselines,
        output_file=output_file
    )


if __name__ == '__main__':
    main(
        image_path=f"{Path(__file__).parent.absolute()}/../../../"
                   f"data/images/Koelnische_Zeitung_1924 - 0085.jpg",
        layout_xml_path=f"{Path(__file__).parent.absolute()}/../../../"
                        f"data/pero_lines_bonn_regions/Koelnische_Zeitung_1924 - 0085.xml",
        output_file=f"{Path(__file__).parent.absolute()}/../../../"
                    f"data/predictionExample.xml'"
    )
