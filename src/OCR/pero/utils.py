from typing import List

import torch
import torch.nn.functional as F


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
