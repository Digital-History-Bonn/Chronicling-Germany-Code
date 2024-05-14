"""Prediction script for Pero baseline detection."""

from pathlib import Path
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
from shapely import Polygon, LineString
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from PIL import ImageDraw, Image
import cv2
from bs4 import BeautifulSoup

from skimage import draw, io
import shapely.geometry as sg

from monai.networks.nets import BasicUNet

from src.baseline_detection.pero.preprocess import get_tag
from src.baseline_detection.pero import layout_helpers as helpers
from src.baseline_detection.xml_conversion import add_baselines


def nonmaxima_suppression(input, element_size=(7, 1)):
    """
    Function from
    https://github.com/DCGM/pero-ocr/blob/master/pero_ocr/layout_engines/cnn_layout_engine.py
    Vertical non-maxima suppression.
    :param input: input array
    :param element_size: structure element for greyscale dilations
    """
    if len(input.shape) == 3:
        dilated = np.zeros_like(input)
        for i in range(input.shape[0]):
            dilated[i, :, :] = ndimage.grey_dilation(
                input[i, :, :], size=element_size)
    else:
        dilated = ndimage.grey_dilation(input, size=element_size)

    return input * (input == dilated)


def plot_lines_on_image(image: torch.Tensor,
                        baselines: List[torch.Tensor],
                        textlines: List[torch.Tensor]):
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
        plt.text(line[0][0], line[0][1], i, fontsize=1, color='red')

    plt.axis('off')  # Turn off axis
    plt.tight_layout()
    plt.savefig(f"{Path(__file__).parent.absolute()}/../../../"
                f"data/PeroBaselinePrediction7.png", dpi=1000)
    print(f"saved fig to {Path(__file__).parent.absolute()}/../../../"
          f"data/PeroBaselinePrediction7.png")


def clustered_lines_to_polygons(t_list, clusters_array):
    """
    Function from https://github.com/DCGM/pero-ocr/blob/master/pero_ocr/layout_engines/cnn_layout_engine.py.
    """
    regions_textlines_tmp = []
    polygons_tmp = []
    for i in range(np.amax(clusters_array) + 1):
        region_textlines = []
        for textline, cluster in zip(t_list, clusters_array):
            if cluster == i:
                region_textlines.append(textline)

        region_poly = helpers.region_from_textlines(region_textlines)
        regions_textlines_tmp.append(region_textlines)
        polygons_tmp.append(region_poly)

    # remove overlaps while minimizing textline modifications
    polygons_tmp = filter_polygons(
        polygons_tmp, regions_textlines_tmp)
    # up to this point, polygons can be any geometry that comes from alpha_shape
    p_list = []
    for region_poly in polygons_tmp:
        if region_poly.is_empty:
            continue
        if region_poly.geom_type == 'MultiPolygon':
            for poly in region_poly.geoms:
                if not poly.is_empty:
                    p_list.append(poly.simplify(5))
        if region_poly.geom_type == 'Polygon':
            p_list.append(region_poly.simplify(5))
    return [np.array(poly.exterior.coords) for poly in p_list]


def filter_polygons(polygons, region_textlines):
    polygons = [helpers.check_polygon(polygon) for polygon in polygons]
    inds_to_remove = []
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            # first check if a polygon is completely inside another, remove the smaller in that case
            if polygons[i].contains(polygons[j]):
                inds_to_remove.append(j)
            elif polygons[j].contains(polygons[i]):
                inds_to_remove.append(i)
            elif polygons[i].intersects(polygons[j]):
                poly_intersection = polygons[i].intersection(polygons[j])
                # remove the overlap from both regions
                poly_tmp = deepcopy(polygons[i])
                polygons[i] = polygons[i].difference(polygons[j])
                polygons[j] = polygons[j].difference(poly_tmp)
                # append the overlap to the one with more textlines in the overlap area
                score_i = 0
                for line in region_textlines[i]:
                    line_poly = helpers.check_polygon(sg.Polygon(line))
                    score_i += line_poly.intersection(poly_intersection).area
                score_j = 0
                for line in region_textlines[j]:
                    line_poly = helpers.check_polygon(sg.Polygon(line))
                    score_j += line_poly.intersection(poly_intersection).area
                if score_i > score_j:
                    polygons[i] = polygons[i].union(poly_intersection)
                else:
                    polygons[j] = polygons[j].union(poly_intersection)
    return [polygon for i, polygon in enumerate(polygons) if i not in inds_to_remove]


class BaselineEngine:
    def __init__(self, model_name: str,
                 cuda: int = 0,
                 downsample: float = 1.0,
                 smooth_line_predictions: bool = True,
                 line_end_weight: float = 1.0,
                 line_detection_threshold: float = 0.2,
                 vertical_line_connection_range: int = 5,
                 paragraph_line_threshold: float = 0.3
                 ):

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

    def preprocess_image(self, image: torch.Tensor, mask_regions: List[torch.Tensor]):
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

    def draw_textregions(self, textregions: List[torch.Tensor], width: int, height: int):
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

    def get_penalty(self, b, shift, x_1, x_2, map, t=1):
        """
        Function from https://github.com/DCGM/pero-ocr/blob/master/pero_ocr/layout_engines/cnn_layout_engine.py
        """
        b_shifted = np.round(b).astype(np.int32)
        b_shifted[:, 1] += int(round(shift))
        x_1_shifted = int(round(x_1)) - np.amin(b_shifted[:, 0])
        x_2_shifted = int(round(x_2)) - np.amin(b_shifted[:, 0])
        map_crop = map[
                   np.clip(np.amin(b_shifted[:, 1] - t), 0, map.shape[0] - 1):
                   np.clip(np.amax(b_shifted[:, 1] + t + 1), 0, map.shape[0] - 1),
                   np.amin(b_shifted[:, 0]):
                   np.amax(b_shifted[:, 0])
                   ]

        b_shifted[:, 1] -= (np.amin(b_shifted[:, 1]) - t)
        b_shifted[:, 0] -= np.amin(b_shifted[:, 0])

        penalty_mask = np.zeros_like(map_crop)
        for b_ind in range(b_shifted.shape[0] - 1):
            try:
                cv2.line(penalty_mask,
                         tuple(b_shifted[b_ind, :]),
                         tuple(b_shifted[b_ind + 1, :]),
                         color=1,
                         thickness=(2 * t) + 1)
            except:
                print("WARNING: Paragraph penalty calculation failed.")
                return 1

        penalty_area = penalty_mask * map_crop

        return np.sum(penalty_area[:, x_1_shifted:x_2_shifted]) / (x_2 - x_1)

    def get_pair_penalty(self, b1, b2, h1, h2, map, ds):
        """
        Function from
        https://github.com/DCGM/pero-ocr/blob/master/pero_ocr/layout_engines/cnn_layout_engine.py.
        """
        x_overlap = max(0, min(np.amax(b1[:, 0]), np.amax(b2[:, 0])) - max(np.amin(b1[:, 0]), np.amin(b2[:, 0])))
        if x_overlap > 5:
            x_1 = int(max(np.amin(b1[:, 0]), np.amin(b2[:, 0])))
            x_2 = int(min(np.amax(b1[:, 0]), np.amax(b2[:, 0])))
            if np.average(b1[:, 1]) > np.average(b2[:, 1]):
                penalty_1 = self.get_penalty(b1 / ds, -h1[0] / ds, x_1 / ds, x_2 / ds, map)
                penalty_2 = self.get_penalty(b2 / ds, h2[1] / ds, x_1 / ds, x_2 / ds, map)
            else:
                penalty_1 = self.get_penalty(b1 / ds, h1[1] / ds, x_1 / ds, x_2 / ds, map)
                penalty_2 = self.get_penalty(b2 / ds, -h2[0] / ds, x_1 / ds, x_2 / ds, map)
            penalty = np.abs(max(penalty_1, penalty_2))
        else:
            penalty = 1
        return penalty

    def make_clusters(self, b_list, h_list, t_list, layout_separator_map):
        """
        Function from https://github.com/DCGM/pero-ocr/blob/master/pero_ocr/layout_engines/cnn_layout_engine.py
        """
        if len(t_list) > 1:

            min_pos = np.zeros([len(t_list), 2], dtype=np.float32)
            max_pos = np.zeros([len(t_list), 2], dtype=np.float32)

            t_list_dilated = []
            for textline, min_, max_ in zip(t_list, min_pos, max_pos):
                textline_poly = sg.Polygon(textline)
                tot_height = np.abs(textline[0, 1] - textline[-1, 1])
                t_list_dilated.append(textline_poly.buffer(3 * tot_height / 4))
                min_[:] = textline.min(axis=0) - tot_height
                max_[:] = textline.max(axis=0) + tot_height

            candidates = np.logical_and(
                np.logical_or(
                    max_pos[:, np.newaxis, 1] <= min_pos[np.newaxis, :, 1],
                    min_pos[:, np.newaxis, 1] >= max_pos[np.newaxis, :, 1]),
                np.logical_or(
                    max_pos[:, np.newaxis, 0] <= min_pos[np.newaxis, :, 0],
                    min_pos[:, np.newaxis, 0] >= max_pos[np.newaxis, :, 0]),
            )
            candidates = np.logical_not(candidates)

            candidates = np.triu(candidates, k=1)
            distances = np.ones((len(t_list), len(t_list)))
            for i, j in zip(*candidates.nonzero()):
                if t_list_dilated[i].intersects(t_list_dilated[j]):
                    penalty = self.get_pair_penalty(
                        b_list[i], b_list[j], h_list[i], h_list[j], layout_separator_map, self.downsample)
                    distances[i, j] = penalty
                    distances[j, i] = penalty

            adjacency = (distances < self.paragraph_line_threshold).astype(int)
            adjacency = adjacency * (1 - np.eye(adjacency.shape[0]))  # put zeros on diagonal
            graph = csr_matrix(adjacency > 0)
            _, clusters_array = connected_components(
                csgraph=graph, directed=False, return_labels=True)

            return clusters_array

        else:
            return [0]

    def parse(self, out_map):
        """
        Function from https://github.com/DCGM/pero-ocr/blob/master/pero_ocr/layout_engines/cnn_layout_engine.py

        Parse input baseline, height and region map into list of baselines
        coords, list of heights and region map
        :param out_map: array of baseline and endpoint probabilities with
        channels: ascender height, descender height, baselines, baseline
        endpoints, region boundaries
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
        baselines_map = (baselines_map - self.line_end_weight * out_map[:, :,
                                                                3]) > self.line_detection_threshold

        # connect vertically disconnected lines - any effect? Parameter is vertical connection distance in pixels.
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
                h_list.append([self.downsample * heights_pred[0], self.downsample * heights_pred[1]])

        # sort lines from LEFT to RIGHT
        x_inds = [np.amin(baseline[:, 0]) + 0.0001 * np.random.rand() for baseline in b_list]
        b_list = [b for _, b in sorted(zip(x_inds, b_list))]
        h_list = [h for _, h in sorted(zip(x_inds, h_list))]

        t_list = [helpers.baseline_to_textline(b, h) for b, h in zip(b_list, h_list)]

        return b_list, h_list, t_list

    def predict(self, image: torch.Tensor, layout: str) -> Tuple[List[Polygon], List[LineString]]:
        """
        Predicts the baselines and textlines on a given image.

        Args:
            image: torch Tensor of image (channel, width, height)
            layout: path to layout xml file
        """
        _, width, height = image.shape
        maps = torch.zeros((5, width, height))  # tensor to save predictions from networks

        # extract layout information
        textregions, mask_regions = self.get_textregions(layout)
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

        b_list, h_list, t_list = helpers.order_lines_vertical(b_list, h_list, t_list)
        # p_list, b_list, t_list = rotate_layout(p_list, b_list, t_list, rot, image.shape)

        baselines = [LineString(line[:, ::-1]).simplify(tolerance=1) for line in b_list]
        textlines = [Polygon(poly[:, ::-1]).simplify(tolerance=1) for poly in t_list]

        return textlines, baselines

    def get_textregions(self, xml_path) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
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
                                           point in coords['points'].split()])[:,
                             torch.tensor([1, 0])]
                textregions.append(textregion)

        return textregions, mask_regions


def main(image_path: str, layout_xml_path: str, output_file: str):
    """
    Predicts textlines and baselines in given image and writes into a annotation xml file.

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
