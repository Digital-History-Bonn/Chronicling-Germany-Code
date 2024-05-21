"""Preprocess for the Newspaper dataset to predict Textlines."""

import glob
import os
from pathlib import Path
from typing import List, Dict, Union, Tuple

import torch
from shapely import LineString
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from skimage import draw
from bs4 import BeautifulSoup, PageElement
from skimage import io
from tqdm import tqdm
import re

from src.baseline_detection.utils import get_bbox, is_valid


def create_baseline_target(shape: Tuple[int, int],
                           baselines: List[torch.Tensor],
                           mask_regions: List[torch.Tensor]) -> np.ndarray:
    """
    Creates a baseline target for given shape.

    Args:
        shape: shape of target
        baselines: list of baselines
        mask_regions: regions to mask in input image,
                      because we know there are no baseline from layout

    Returns:
        numpy array representing target for baseline prediction
    """
    # Create a blank target filled with ones (white)
    target = np.zeros((*shape, 2), dtype=np.uint8)
    target[:, :, 1] = 1  # mask with default value true

    # create PIL draw
    baseline_img = Image.fromarray(target[:, :, 0])
    baseline_draw = ImageDraw.Draw(baseline_img)

    # Draw baselines
    for baseline in baselines:
        line = LineString(torch.flip(baseline, dims=[1]))
        baseline_draw.line(line.coords, fill=1, width=1)

    target[:, :, 0] = np.array(baseline_img)

    for mask_region in mask_regions:
        # draw mask to remove not text regions
        if len(mask_region) >= 3:
            rr, cc = draw.polygon(mask_region[:, 1], mask_region[:, 0], shape=shape)
            target[rr, cc, 1] = 0

    return np.array(target)


def get_tag(textregion: PageElement) -> str:
    """
    Returns the tag of the given textregion.

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


def get_reading_order_idx(textregion: PageElement) -> int:
    """
    Extracts reading order from textregion PageElement.

    Args:
        textregion: PageElement of Textregion

    Returns:
         Reading Order Index as int
    """
    desc = textregion['custom']
    match = re.search(r"readingOrder\s*\{index:(\d+);\}", desc)
    if match is None:
        return -1
    return int(match.group(1))


def extract(xml_path: str
            ) -> Tuple[
                       List[Dict[str, Union[torch.Tensor, List[torch.Tensor], int]]],
                       List[torch.Tensor]]:
    """
    Extracts the annotation from the xml file.

    Args:
        xml_path: path to the xml file.

    Returns:
        A list of dictionary representing all Textregions in the given document
        A list of polygons as torch tensors for masking areas
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
            part = torch.tensor([tuple(map(int, point.split(','))) for
                                 point in coords['points'].split()])
            mask_regions.append(part)

        if tag in ['heading', 'article_', 'caption', 'paragraph']:
            coords = region.find('Coords')
            part = torch.tensor([tuple(map(int, point.split(','))) for
                                 point in coords['points'].split()])[:, torch.tensor([1, 0])]
            part = torch.tensor(get_bbox(part))

            region_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor], int]] = {
                'part': part,
                'bboxes': [],
                'masks': [],
                'baselines': [],
                'readingOrder': get_reading_order_idx(region)}

            text_region = region.find_all('TextLine')
            for text_line in text_region:
                polygon = text_line.find('Coords')
                baseline = text_line.find('Baseline')
                if baseline:
                    # get and shift baseline
                    line = torch.tensor([tuple(map(int, point.split(','))) for
                                         point in baseline['points'].split()])
                    line = line[:, torch.tensor([1, 0])]

                    line -= part[:2].unsqueeze(0)

                    region_dict['baselines'].append(line)  # type: ignore

                    # get mask
                    polygon_pt = torch.tensor([tuple(map(int, point.split(','))) for
                                               point in polygon['points'].split()])
                    polygon_pt = polygon_pt[:, torch.tensor([1, 0])]

                    # move mask to be in subimage
                    polygon_pt -= part[:2].unsqueeze(0)

                    # calc bbox for line
                    box = torch.tensor(get_bbox(polygon_pt))[torch.tensor([1, 0, 3, 2])]
                    box = box.clip(min=0)

                    # add bbox to data
                    if is_valid(box):
                        region_dict['bboxes'].append(box)  # type: ignore

                        # add mask to data
                        region_dict['masks'].append(polygon_pt)  # type: ignore

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


def main(image_folder: str, target_folder: str, output_path: str) -> None:
    """
    Preprocesses the complete dataset so it can be used for training.

    Args:
        image_folder: path to images
        target_folder: path to xml files
        output_path: path to save folder
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
        try:
            os.makedirs(f"{output_path}/{document_name}/", exist_ok=False)
        except OSError:
            continue
        regions, mask_regions = extract(tar_path)

        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)  # type: ignore
        torch_image = to_tensor(image).permute(1, 2, 0).to(torch.uint8)

        for i, region in enumerate(regions):
            # create dict for subimage
            os.makedirs(f"{output_path}/{document_name}/region_{i}", exist_ok=True)

            # save subimage
            subimage = torch_image[region['part'][0]: region['part'][2],    # type: ignore
                                   region['part'][1]: region['part'][3]]    # type: ignore

            io.imsave(f"{output_path}/{document_name}/region_{i}/image.jpg",
                      subimage.to(torch.uint8))

            # save target information
            torch.save(region['masks'],
                       f"{output_path}/{document_name}/region_{i}/masks.pt")
            torch.save(region['baselines'],
                       f"{output_path}/{document_name}/region_{i}/baselines.pt")
            torch.save(region['bboxes'],
                       f"{output_path}/{document_name}/region_{i}/bboxes.pt")

            target = create_baseline_target(subimage.shape[:2],
                                            region['baselines'],    # type: ignore
                                            region['masks'])        # type: ignore
            np.savez_compressed(f"{output_path}/{document_name}/region_{i}/baselines",
                                array=target)


if __name__ == "__main__":
    main(f'{Path(__file__).parent.absolute()}/../../../data/images',
         f'{Path(__file__).parent.absolute()}/../../../data/pero_lines_bonn_regions',
         f'{Path(__file__).parent.absolute()}/../../../data/preprocessed')

    # test_array = np.load(
    #     f'{Path(__file__).parent.absolute()}/../../../'
    #     f'data/Newspaper/preprocessed/Koelnische_Zeitung_1924 - 0008/baselines.npz')['array']
    # image = io.imread(
    #    f'{Path(__file__).parent.absolute()}/../../../'
    #    f'data/Newspaper/newspaper-dataset-main-images/images/Koelnische_Zeitung_1924 - 0008.jpg')
