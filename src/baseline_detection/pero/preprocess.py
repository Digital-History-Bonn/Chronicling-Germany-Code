"""Preprocess for the Newspaper dataset to predict Textlines."""

import glob
import os
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional

import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from skimage import draw
from bs4 import BeautifulSoup, PageElement
from matplotlib import pyplot as plt
from shapely.ops import split
from shapely.geometry import LineString, Polygon
from tqdm import tqdm
import re


def get_bbox(
        points: Union[np.ndarray, torch.Tensor],  # type: ignore
        corners: Union[None, List[int]] = None,
        tablebbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[int, int, int, int]:
    """
    Creates a bounding box around all given points.

    Args:
        points: np.ndarray of shape (N x 2) containing a list of points
        corners: corners can be defined if this is the case only the corner points are used for bb
        tablebbox: if given, bbox is calculated relative to table

    Returns:
        coordinates of bounding box in the format (x_min, y_min, x_max, y_max)

    """
    if corners:
        points = points[corners]

    x_max, x_min = points[:, 0].max(), points[:, 0].min()
    y_max, y_min = points[:, 1].max(), points[:, 1].min()
    # swap 0 and 1 for tablebox if np.flip in extract_glosat_annotation
    if tablebbox:
        x_min -= tablebbox[0]
        y_min -= tablebbox[1]
        x_max -= tablebbox[0]
        y_max -= tablebbox[1]

    return x_min, y_min, x_max, y_max  # type: ignore


def split_textbox(box: Polygon, baseline: LineString):
    # extend line to avoid not completely intersecting polygone
    points = list(baseline.coords)
    new_coords = [(-100, points[0][1])] + points + [(points[-1][0] + 100, points[-1][1])]
    baseline = LineString(new_coords)

    # Use the split method to split the polygon
    parts = split(box, baseline)

    if len(parts.geoms) >= 2:
        # Determine which part is above and below the split line
        ascender = parts.geoms[0] if parts.geoms[0].centroid.y < parts.geoms[1].centroid.y else \
            parts.geoms[1]
        descender = parts.geoms[1] if parts.geoms[0].centroid.y < parts.geoms[1].centroid.y else \
            parts.geoms[0]
        return ascender, descender

    else:
        raise ValueError('Baseline and polygone not intersecting!')


def is_valid(box: torch.Tensor):
    """
    Checks if given bounding box has a valid size.

    Args:
        box: bounding box (xmin, ymin, xmax, ymax)

    Returns:
        True if bounding box is valid
    """
    if box[2] - box[0] <= 0:
        return False
    if box[3] - box[1] <= 0:
        return False
    return True


def calc_heights(polygon: Polygon, baseline: torch.tensor):
    bbox = polygon.bounds

    # Move the coordinates relative to the bottom-left corner of its bounding box
    relative_polygon = np.array(polygon.exterior.coords) - np.array(bbox[:2])

    # calc width and hight of polygon image
    shape = (int(bbox[3] - bbox[1]), int(bbox[2] - bbox[0]))

    # draw polygon
    polygon_image = torch.zeros(shape, dtype=int)
    rr, cc = draw.polygon(relative_polygon[:, 1], relative_polygon[:, 0], shape=shape)
    polygon_image[rr, cc] = 1

    x_coords = []
    y_coords = []
    for i in range(len(baseline) - 1):
        rr, cc = draw.line(baseline[i, 0],
                           baseline[i, 1],
                           baseline[i + 1, 0],
                           baseline[i + 1, 1])
        x_coords.extend(rr)
        y_coords.extend(cc)

    x_coords = torch.tensor(x_coords)
    y_coords = torch.tensor(y_coords)

    indices = (y_coords - torch.tensor([bbox[0]])).long().clip(min=0, max=shape[1] - 1)
    values = torch.sum(polygon_image, dim=0)[indices]

    return x_coords, y_coords, values


def draw_baseline_target(shape: Tuple[int, int],
                         baselines: List[torch.Tensor],
                         textlines: List[torch.Tensor],
                         mask_regions: List[torch.Tensor],
                         textregions: List[List[torch.Tensor]],
                         name: str,
                         width: int = 3) -> np.ndarray:
    """
    Draw baseline target for given shape.

    Args:
        shape: shape of target
        baselines: list of baselines
        textlines: list of polygons around the textline
        mask_regions: list of polygons to mask on image
        textregions: list of polygons around paragraph
        name: filename for logging errors
        width: width of the drawn baselines
    """
    # Create a blank target filled with ones (white)
    target = np.zeros((*shape, 6), dtype=np.uint8)
    target[:, :, 5] = 1  # mask with default value true

    # create PILDraw instances to draw baselines, textregions and limiters
    limiter_img = Image.fromarray(target[:, :, 3])
    limiter_draw = ImageDraw.Draw(limiter_img)

    textregion_img = Image.fromarray(target[:, :, 4])
    textregion_draw = ImageDraw.Draw(textregion_img)

    baseline_img = Image.fromarray(target[:, :, 2])
    baseline_draw = ImageDraw.Draw(baseline_img)

    # Draw targets
    for baseline, textline in zip(baselines, textlines):
        # calc ascender and descender
        line = LineString(torch.flip(baseline, dims=[1]))
        polygon = Polygon(torch.flip(textline, dims=[1]))

        # draw limiter
        min_x, min_y, max_x, max_y = polygon.bounds
        limiter_draw.line([(min_x, min_y), (min_x, max_y)], fill=1, width=width)
        limiter_draw.line([(max_x, min_y), (max_x, max_y)], fill=1, width=width)

        # draw baseline
        baseline_draw.line(line.coords, fill=1, width=1)

        try:
            ascender, descender = split_textbox(polygon, line)
        except ValueError as e:
            print(f"{e} in image {name}\n")
            continue
        # draw ascender
        x_coords, y_coords, values = calc_heights(ascender, baseline)
        x_coords = x_coords[y_coords < shape[1]]
        values = values[y_coords < shape[1]]
        y_coords = y_coords[y_coords < shape[1]]

        target[x_coords, y_coords, 0] = values

        # draw descender
        x_coords, y_coords, values = calc_heights(descender, baseline)
        x_coords = x_coords[y_coords < shape[1]]
        values = values[y_coords < shape[1]]
        y_coords = y_coords[y_coords < shape[1]]

        target[x_coords, y_coords, 1] = values

    target[:, :, 3] = np.array(limiter_img)
    target[:, :, 2] = np.array(baseline_img)

    # draw textregions
    for textregion in textregions:
        textregion_draw.polygon([(x[1], x[0]) for x in textregion], fill=0, outline=1, width=width)

    target[:, :, 4] = np.array(textregion_img)

    # draw masks
    for mask_region in mask_regions:
        if len(mask_region) >= 3:
            rr, cc = draw.polygon(mask_region[:, 1], mask_region[:, 0], shape=shape)
            target[rr, cc, 5] = 0

    return np.array(target)


def get_tag(textregion: PageElement):
    """
    Returns the tag of the given textregion

    Args:
        textregion: PageElement of Textregion

    Returns:
        Given tag of that Textregion
    """
    desc = textregion['custom']
    match = re.search(r"\{type:.*;\}", desc)
    if match is None:
        return 'UnknownRegion'
    return match.group()[6:-2]


def extract(xml_path: str, create_subimages: bool) -> Tuple[List[Dict[str, List[torch.Tensor]]], List[torch.Tensor]]:
    """
    Extracts the annotation from the xml file.

    Args:
        xml_path: path to the xml file.
        create_subimages: create subimages based on the paragraph segmentation

    Returns:
        A list of dictionary representing all Textregions in the given document
    """
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    paragraphs = []
    mask_regions = []

    text_regions = page.find_all('TextRegion')
    for region in text_regions:
        tag = get_tag(region)

        if tag in ['table', 'header']:
            coords = region.find('Coords')
            bbox = torch.tensor([tuple(map(int, point.split(','))) for
                                 point in coords['points'].split()])
            mask_regions.append(bbox)

        if tag in ['heading', 'article_', 'caption', 'paragraph']:
            coords = region.find('Coords')
            textregion = torch.tensor([tuple(map(int, point.split(','))) for
                                       point in coords['points'].split()])[:, torch.tensor([1, 0])]
            bbox = torch.tensor(get_bbox(textregion))

            region_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {'part': bbox,
                                                                               'textregion': textregion,
                                                                               'bboxes': [],
                                                                               'masks': [],
                                                                               'baselines': []}

            text_region = region.find_all('TextLine')
            for text_line in text_region:
                polygon = text_line.find('Coords')
                baseline = text_line.find('Baseline')
                if baseline:
                    # get and shift baseline
                    line = torch.tensor([tuple(map(int, point.split(','))) for
                                         point in baseline['points'].split()])[:, torch.tensor([1, 0])]

                    if create_subimages:
                        line -= bbox[:2].unsqueeze(0)

                    region_dict['baselines'].append(line)  # type: ignore

                    # get mask
                    polygon_pt = torch.tensor([tuple(map(int, point.split(','))) for
                                               point in polygon['points'].split()])[:,
                                 torch.tensor([1, 0])]

                    # move mask to be in subimage
                    if create_subimages:
                        polygon_pt -= bbox[:2].unsqueeze(0)

                    # calc bbox for line
                    box = torch.tensor(get_bbox(polygon_pt))[torch.tensor([1, 0, 3, 2])]
                    box = box.clip(min=0)

                    # add bbox to data
                    if is_valid(box):
                        region_dict['bboxes'].append(box)  # type: ignore

                        # add mask to data
                        region_dict['masks'].append(polygon_pt)  # type: ignore

            if region_dict['bboxes']:
                region_dict['bboxes'] = torch.stack(region_dict['bboxes'])  # type: ignore
                paragraphs.append(region_dict)

    return paragraphs, mask_regions


def rename_files(folder_path: str) -> None:
    """
    Renames all files and folders in given folder by replacing 'ö' with 'oe'.

    Args:
        folder_path: path to folder
    """
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file
    for filename in files:
        # Check if the file name contains 'ö'
        if 'ö' in filename:
            # Replace 'ö' with 'oe' in the filename
            new_filename = filename.replace('ö', 'oe')

            # Construct the full old and new paths
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed {filename} to {new_filename}")


def plot_target(image: np.ndarray,
                target: np.ndarray, figsize=(50, 10), dpi=500):
    """
    Creates and saves an Image of the targets on the image

    Args:
        image: numpy array representation of the image (width, height, channel)
        target numpy array representation of the target (width, height, channel)
        figsize: size of the plot (default: 50, 10)
        dpi: dpi of the plot
    """
    # Plot the first image
    attributes = ['ascenders', 'descenders', 'baselines', 'marker', 'textregion']
    image = image * target[:, :, 5, None]

    # image = image[500:1500, 500:1500]
    # target = target[500:1500, 500:1500]

    fig, axes = plt.subplots(1, len(attributes), figsize=figsize)  # Adjust figsize as needed

    print(f"{target.shape=}")
    for i, attribute in enumerate(attributes):
        axes[i].imshow(image.astype(np.uint8))
        if attribute in ['ascenders', 'descenders']:
            axes[i].imshow(target[:, :, i].astype(np.uint8) * 4, alpha=0.5)
        else:
            axes[i].imshow(target[:, :, i].astype(np.uint8), cmap='gray', alpha=0.5)
        axes[i].set_title(attribute, fontsize=26)
        axes[i].axis('off')

    plt.tight_layout()  # Adjust layout
    plt.subplots_adjust(wspace=0.05)  # Adjust space between subplots

    # Display the plot with higher DPI
    plt.savefig(f'{Path(__file__).parent.absolute()}/../../../data/assets/TargetExample.png',
                dpi=dpi)
    plt.show()


def main(image_folder: str, target_folder: str, output_path: str, create_baseline_target: bool,
         create_subimages: bool) -> None:
    """
    Preprocesses the complete dataset so it can be used for training.

    Args:
        image_folder: path to images
        target_folder: path to xml files
        output_path: path to save folder
        create_baseline_target: if True creates targets for baseline UNet
    """
    to_tensor = transforms.PILToTensor()

    target_paths = [x for x in glob.glob(f"{target_folder}/*.xml")]
    image_paths = [f"{image_folder}/{x.split(os.sep)[-1][:-4]}.jpg" for x in target_paths]

    print(f"{len(image_paths)=}")
    print(f"{len(target_paths)=}")

    for img_path, tar_path in tqdm(zip(image_paths, target_paths),
                                   total=len(image_paths),
                                   desc='preprocessing'):
        document_name = img_path.split(os.sep)[-1][:-4]

        # check if folder already exists if yes skip this image
        if os.path.exists(f"{output_path}/target_{document_name}.npz"):
            continue

        # extract annotation information from annotation xml file
        regions, mask_regions = extract(tar_path, create_subimages=create_subimages)

        # open image, check orientation, convert to tensor
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)
        image = to_tensor(image).permute(1, 2, 0).to(torch.uint8)

        # create a list for masks, baselines, bboxes and textregion information
        masks, baselines, bboxes, textregion = [], [], [], []
        for region in regions:
            # save target information
            masks.extend(region['masks'])
            baselines.extend(region['baselines'])
            bboxes.extend(region['bboxes'])
            textregion.append(region['textregion'])

        # create target as numpy array and save it in a compressed file
        target = draw_baseline_target(image.shape[:2], baselines, masks, mask_regions, textregion, document_name)
        np.savez_compressed(f"{output_path}/{document_name}",
                            array=target)


if __name__ == "__main__":
    main(f'{Path(__file__).parent.absolute()}/../../../data/images',
         f'{Path(__file__).parent.absolute()}/../../../data/pero_lines_bonn_regions',
         f'{Path(__file__).parent.absolute()}/../../../data/preprocessed',
         create_baseline_target=True,
         create_subimages=False)

    # test_array = np.load(
    #     f'{Path(__file__).parent.absolute()}/../../data/Newspaper/preprocessed3/Koelnische Zeitung 1866.06-1866.09 - 0182/baselines.npz')[
    #     'array']
    # image = io.imread(
    #     f'{Path(__file__).parent.absolute()}/../../data/Newspaper/newspaper-dataset-main-images/images/Koelnische Zeitung 1866.06-1866.09 - 0182.jpg')
    # plot_target(image, test_array)