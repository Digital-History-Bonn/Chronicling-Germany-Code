from typing import List, Tuple

import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup

from src.OCR.utils import get_bbox


class Tokenizer:
    def __init__(self, alphabet: List[str],
                 pad: bool=False,
                 max_lenght: int = 512,
                 print_nan: bool = False):
        self.alphabet = {token: i for i, token in enumerate(alphabet)}
        self.inverse = {i: token for i, token in enumerate(alphabet)}
        self.pad = pad
        self.max_lenght = max_lenght
        self.print_nan = print_nan

    def __call__(self, text: str) -> torch.Tensor:
        ids = [self.alphabet[t] if t in self.alphabet.keys() else self.alphabet['<NAN>'] for t in text]
        ids.insert(0, self.alphabet['<START>'])
        ids.append(self.alphabet['<END>'])
        if self.pad:
            ids = F.pad(torch.tensor(ids), pad=(0, self.max_lenght - len(ids)), mode='constant', value=self.alphabet['<PAD>'])

        return torch.tensor(ids[:self.max_lenght], dtype=torch.long)

    def to_text(self, ids: torch.Tensor) -> str:
        if ids.dim() != 1: raise ValueError('ids must have dimension 1')

        text = ''
        for id in ids[1:]:
            token = self.inverse[id.item()]
            if token == '<END>':
                break
            text += token if self.print_nan or token != '<NAN>' else ' '

        return text


def read_xml(xml_path: str) -> Tuple[List[torch.Tensor], List[str]]:
    with open(xml_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    bboxes = []
    texts = []

    text_lines = page.find_all('TextLine')
    for line in text_lines:
        coords = line.find('Coords')
        region_polygon = torch.tensor([tuple(map(int, point.split(','))) for
                                       point in coords['points'].split()])
        bboxes.append(torch.tensor(get_bbox(region_polygon)))
        texts.append(line.find('Unicode').text)

    return bboxes, texts