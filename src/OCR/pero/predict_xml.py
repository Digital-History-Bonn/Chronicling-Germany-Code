import argparse
import glob
import json
import os
from typing import Tuple, List

import torch
import torch.nn.functional as F
from skimage import io

from bs4 import BeautifulSoup, ResultSet
from tqdm import tqdm, trange

from src.OCR.pero.config import ALPHABET, PAD_HEIGHT, PAD_WIDTH
from src.OCR.pero.ocr_engine import transformer
from src.OCR.pero.ocr_engine.transformer import TransformerOCR
from src.OCR.pero.predict import predict
from src.OCR.pero.tokenizer import Tokenizer
from src.OCR.utils import get_bbox


def read_xml(soup: BeautifulSoup) -> Tuple[ResultSet, List[torch.Tensor]]:
    """
    Reads out textlines and corresponding bounding boxes form soup.
    Args:
        soup: soup object with textline annotations.

    Returns:
        text_lines: ResultSet with textlines
        bboxes: Bounding boxes from textlines.
    """
    page = soup.find('Page')
    bboxes = []

    text_lines = page.find_all('TextLine')
    for line in text_lines:
        coords = line.find('Coords')
        region_polygon = torch.tensor([tuple(map(int, point.split(','))) for
                                       point in coords['points'].split()])
        bboxes.append(torch.tensor(get_bbox(region_polygon)))

    return text_lines, bboxes


def replace_text(soup, new_text):
    # Find the first occurrence of the tag
    tag = soup.find('Unicode')

    if tag and tag.text:
        # Replace the text inside the tag
        tag.string.replace_with(new_text)


def predict_and_write(model: TransformerOCR,
                      tokenizer: Tokenizer,
                      device: torch.device,
                      image_path: str,
                      anno_path: str,
                      out_path: str):
    """
    Predicts textlines based on annotated lines and writes them into the xml file.

    Args:
        model: model to use for prediction
        tokenizer: tokenizer to use for prediction
        device: device to use for prediction
        image_path: path to image
        anno_path: path to annotation
        out_path: path to save the resulting xml file
    """
    if os.path.exists(out_path):
        return

    image = torch.tensor(io.imread(image_path)).permute(2, 0, 1).to(device)

    with open(anno_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    textlines, bboxes = read_xml(soup)

    for line, bbox in tqdm(zip(textlines, bboxes), total=len(textlines), desc="precessing lines",
                           disable=True):
        crop = image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

        pad_height = max(0, PAD_HEIGHT - crop.shape[1])
        pad_width = max(0, PAD_WIDTH - crop.shape[2])
        crop = F.pad(crop, (pad_width, 0, pad_height, 0), "constant", 0)
        crop = crop[:, :PAD_HEIGHT]

        text = predict(model, tokenizer, crop[None].float() / 255)

        replace_text(line, text)

    # save results
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
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
    parser = argparse.ArgumentParser(description="train Pero OCR")

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="model",
        help="Name of the model and the log files."
    )

    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="path for folder with images jpg files to predict."
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="path to output folder for predicted xml files."
    )

    return parser.parse_args()


def main():
    args = get_args()

    print(f"{torch.cuda.is_available()=}")
    device = (
        torch.device(f"cuda:{0}")
        if torch.cuda.is_available() and 0 >= 0
        else torch.device("cpu")
    )
    print(f"using {device}")

    with open("src/OCR/pero/config.json", "r") as file:
        json_data = json.load(file)

    model: transformer.TransformerOCR = transformer.build_net(net=json_data,
                                                              input_height=PAD_HEIGHT,
                                                              input_channels=3,
                                                              nb_output_symbols=len(ALPHABET) - 2)
    model.load_state_dict(torch.load(f"models/{args.model}.pt"))
    model.to(device)

    tokenizer = Tokenizer(ALPHABET)

    images = glob.glob(f"{args.data}/*.jpg")
    annos = [f"{x[:-4]}.xml" for x in images]
    outputs = [f"{args.output}/{os.path.basename(x)[:-4]}.xml" for x in images]

    for i in trange(len(images), desc="preprocessing images"):
        predict_and_write(model,
                          tokenizer,
                          device,
                          image_path=images[i],
                          anno_path=annos[i],
                          out_path=outputs[i])


if __name__ == '__main__':
    main()
