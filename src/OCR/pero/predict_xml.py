import glob
import json
import os
from typing import Tuple, List

import torch
import torch.nn.functional as F
from skimage import io

from bs4 import BeautifulSoup, ResultSet
from tqdm import tqdm, trange

from src.OCR.pero.ocr_engine import transformer
from src.OCR.pero.predict import predict
from src.OCR.pero.trainer import CROP_HEIGHT
from src.OCR.pero.utils import Tokenizer
from src.OCR.utils import get_bbox


ALPHABET = ['<PAD>', '<START>', '<NAN>', '<END>',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'ä', 'ö', 'ü', 'ſ', 'ẞ', 'à', 'è',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            ' ', ',', '.', '?', '!', '-', '_', ':', ';', '/', '\\', '(', ')', '[', ']', '{', '}', '%', '$',
            '\"', '„', '“', '\'', '’', '&', '+', '~']


def read_xml(soup: BeautifulSoup) -> Tuple[ResultSet, List[torch.Tensor]]:
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


def compare(model, tokenizer, device, image_path: str, anno_path: str, out_path: str):
    image = torch.tensor(io.imread(image_path)).permute(2, 0, 1).to(device)

    with open(anno_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    textlines, bboxes = read_xml(soup)

    for line, bbox in tqdm(zip(textlines, bboxes), total=len(textlines), desc="precessing lines", disable=True):
        crop = image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

        pad_height = max(0, 64 - crop.shape[1])
        pad_width = max(0, 16 - crop.shape[2])
        crop = F.pad(crop, (pad_width, 0, pad_height, 0), "constant", 0)
        crop = crop[:, :64]

        text = predict(model, tokenizer, crop[None].float() / 255)

        replace_text(line, text)

    # save results
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as file:
        file.write(soup.prettify()
                   .replace("<Unicode>\n      ", "<Unicode>")
                   .replace("\n     </Unicode>", "</Unicode>"))

def main():
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
                                                              input_height=CROP_HEIGHT,
                                                              input_channels=3,
                                                              nb_output_symbols=len(ALPHABET) - 2)
    model.load_state_dict(torch.load(f"models/{'testBender2_es'}.pt"))
    model.to(device)

    tokenizer = Tokenizer(ALPHABET)

    images = glob.glob("data/preprocessedOCR/test/*.jpg")
    annos = [f"{x[:-4]}.xml" for x in images]
    outputs = [f"logs/peroPredict/test/{os.path.basename(x)[:-4]}.xml" for x in images]

    for i in trange(len(images), desc="preprocessing images"):
        compare(model,
                tokenizer,
                device,
                image_path=images[i],
                anno_path=annos[i],
                out_path=outputs[i])

if __name__ == '__main__':
    main()
