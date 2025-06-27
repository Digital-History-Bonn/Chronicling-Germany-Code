import argparse
import glob
import json
import os
import shutil
from os.path import basename

import matplotlib.pyplot as plt
import yaml
from bs4 import BeautifulSoup
from shapely.geometry import Polygon
from skimage import io
from tqdm import tqdm

from cgprocess.OCR.shared.utils import adjust_path

LABEL_ASSIGNMENTS = {
    "caption": 0,
    "table": 1,
    "article": 2,
    "article_": 2,
    "paragraph": 2,
    "heading": 3,
    "header": 4,
    "image": 5,
    "Image": 5,
    "inverted_text": 6
}

CLASS_ASSIGNMENTS = {
    0: "caption",
    1: "table",
    2: "article",
    3: "heading",
    4: "header",
    5: "image",
    6 : "inverted_text"
}
colors = ['tab:blue', 'tab:orange', 'tab:green',
          'tab:red', 'tab:purple', 'tab:pink',
          'tab:brown', 'tab:cyan', 'tab:gray']


def plot_yolo_boxes(yolo_data: str, image_path: str) -> None:
    """
    Plots YOLO bounding boxes on an image.

    Args:
        yolo_data (str): YOLO-format string with bounding boxes.
        image_path (str): Path to the image on which to draw bounding boxes.

    Returns:
        None: Displays the image with bounding boxes.
    """
    # Load the image using skimage
    image = io.imread(image_path)
    img_height, img_width = image.shape[:2]
    print(img_height, img_width)

    # Parse the YOLO data string
    boxes = []
    for line in yolo_data.strip().splitlines():
        class_id, x_center, y_center, width, height = map(float, line.split())
        # Convert normalized coordinates back to pixel values
        x_center *= img_width
        y_center *= img_height
        box_width = width * img_width
        box_height = height * img_height

        # Calculate top-left and bottom-right corners of the bounding box
        x_min = int(x_center - box_width / 2)
        y_min = int(y_center - box_height / 2)
        x_max = int(x_center + box_width / 2)
        y_max = int(y_center + box_height / 2)

        boxes.append((class_id, x_min, y_min, x_max, y_max))

    # Plot the image and overlay bounding boxes
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    ax = plt.gca()

    # Draw each bounding box
    for class_id, x_min, y_min, x_max, y_max in boxes:
        # Create a rectangle patch
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=1, edgecolor=colors[int(class_id)], facecolor='none')
        ax.add_patch(rect)
        # Add class ID text above the bounding box
        plt.text(x_min, y_min - 5, f"{CLASS_ASSIGNMENTS[class_id]}",
                 color='black', fontsize=6, backgroundcolor=None)

    plt.axis('off')
    plt.show()


def read_xml(path: str) -> tuple:
    """
    Reads out polygon information and classes from xml file.

    Args:
        path (str): path to xml file

    Returns:
        polygons (list): list of polygons
        classes (list): list of classes
        width (int): width of page
        height (int): height of page
    """
    with open(path, "r", encoding="utf-8") as file:
        data = file.read()

    soup = BeautifulSoup(data, "xml")
    page = soup.find('Page')
    width, height = int(page['imageHeight']), int(page['imageWidth'])

    polygons = []
    classes = []

    # Loop through each relevant region tag
    for region in soup.find_all(['TextRegion', 'TableRegion', 'GraphicRegion']):
        # Extract the points for the polygon
        coords = region.find("Coords")
        custom = region.get("custom", "")
        if coords and 'structure' in custom:
            points_str = coords['points']
            # Convert the points string to a list of (x, y) tuples
            points = [(int(x), int(y)) for x, y in (pair.split(",") for pair in points_str.split())]

            # Extract the class information from the 'custom' attribute or region tag name
            structure_type = custom.split("structure {type:")[1].split(";")[0]

            # Create a shapely Polygon from the points
            if len(points) > 2 and structure_type in LABEL_ASSIGNMENTS:
                polygons.append(Polygon(points))
                classes.append(LABEL_ASSIGNMENTS[structure_type])

    return polygons, classes, width, height


def convert_polygon_to_yolo(img_width: int, img_height: int, polygon: Polygon, class_id: int) -> str:
    """
    Converts a polygon to YOLO format string.

    Args:
        img_width (int): Width of the image.
        img_height (int): Height of the image.
        polygon (Polygon): A shapely Polygon object defining the object's shape.
        class_id (int): The class ID of the object.

    Returns:
        str: A formatted string in YOLO format "class_id x_center y_center width height".
    """
    # Get bounding box coordinates from the polygon
    min_x, min_y, max_x, max_y = polygon.bounds

    # Calculate YOLO format coordinates
    x_center = ((min_x + max_x) / 2) / img_height
    y_center = ((min_y + max_y) / 2) / img_width
    width = (max_x - min_x) / img_height
    height = (max_y - min_y) / img_width

    # Format output as a YOLO string
    yolo_format = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

    return yolo_format


def xml_to_yolo(path: str) -> str:
    polygons, classes, width, height = read_xml(path)

    yolo_format = ""
    for polygon, class_id in zip(polygons, classes):
        yolo_format += convert_polygon_to_yolo(width, height, polygon, class_id)

    return yolo_format


def main(annotation_path: str, image_path: str, split_file: str, output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)

    with open(split_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    data['train'] = data.pop('Training')
    data['val'] = data.pop('Validation')
    data['test'] = data.pop('Test')

    split = {value: key for key, values in data.items() for value in values}

    annotations = glob.glob(f"{annotation_path}/*.xml")
    images = [f"{image_path}/{basename(file)[:-4]}.jpg" for file in annotations]

    for annotation, image in tqdm(zip(annotations, images), total=len(images),
                                  desc='processing data'):
        folder = split.get(basename(annotation)[:-4])
        os.makedirs(f"{output_path}/{folder}/images", exist_ok=True)
        os.makedirs(f"{output_path}/{folder}/labels", exist_ok=True)

        # create YOLO format .txt-file
        yolo_format = xml_to_yolo(annotation)
        with open(f"{output_path}/{folder}/labels/{basename(annotation)[:-4]}.txt", 'w',
                  encoding='utf-8') as file:
            file.write(yolo_format)

        # copy image to right location
        shutil.copy(image, f"{output_path}/{folder}/images/{basename(image)}")

    # create .yaml-file
    # Create the base dictionary structure
    dataset_config = {
        'path': output_path,
        'train': "train/images",
        'val': "val/images",
        'names': CLASS_ASSIGNMENTS
    }

    # Write the data to a YAML file
    with open(f"{output_path}/CGD.yaml", 'w', encoding="utf-8") as file:
        yaml.dump(dataset_config, file, default_flow_style=False)


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    # pylint: disable=duplicate-code
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument(
        "--images",
        "-i",
        type=str,
        default=None,
        help="path for folder with images. Need to be jpg."
    )

    parser.add_argument(
        "--annotations",
        "-a",
        type=str,
        default=None,
        help="path for folder with annotation xml files."
    )

    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default=None,
        help="path to the split json."
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="path to the folder where to save the preprocessed files",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    images_dir = adjust_path(args.images)
    annotation_dir = adjust_path(args.annotations)
    output_dir = adjust_path(args.output)

    main(annotation_path=annotation_dir,
         image_path=images_dir,
         split_file=args.split,
         output_path=output_dir)
