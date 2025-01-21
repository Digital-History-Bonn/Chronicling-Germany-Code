"""Tokenizer for Transformer based OCR."""

from typing import List

import torch
import torch.nn.functional as F


class Tokenizer:
    """Tokenizer for Transformer based OCR."""
    def __init__(self, alphabet: List[str],
                 pad: bool=False,
                 max_length: int = 512,
                 print_nan: bool = False):
        """
        Tokenizer for Transformer based OCR.

        Args:
            alphabet: List with all tokens to use for tokenization.
            pad: pads all sequences to max length if true
            max_length: maximum length of tokenized sequences.
            print_nan: should nan token be considered when converting back to text
        """

        self.alphabet = {token: i for i, token in enumerate(alphabet)}
        self.inverse = dict(enumerate(alphabet))
        self.pad = pad
        self.max_length = max_length
        self.print_nan = print_nan

    def __call__(self, text: str) -> torch.Tensor:
        """
        Tokenize a sequence.
        Args:
            text: string

        Returns:
            Torch tensor with token ids.
        """
        ids = [self.__get_token(t) for t in text]
        ids.insert(0, self.alphabet['<START>'])
        ids.append(self.alphabet['<END>'])
        if self.pad:
            ids = F.pad(torch.tensor(ids),                      # type: ignore
                        pad=(0, self.max_length - len(ids)),
                        mode='constant',
                        value=self.alphabet['<PAD>'])

        return torch.tensor(ids[:self.max_length], dtype=torch.long)

    def single_token(self, input: str) -> int:
        """
        Tokenize a single character. This can include returning the index of a start, end or nan token.
        Args:
            input: string

        Returns:
            int with token id.
        """
        return self.__get_token(input)

    def single_token_to_text(self, input_id: int) -> str:
        """
         Converts single token id back to text.
         """
        return self.inverse[input_id]

    def __get_token(self, input: str) -> int:
        """
        Tokenize a single character.
        """
        return self.alphabet[input] if input in self.alphabet.keys() else self.alphabet['<NAN>']

    def to_text(self, token_ids: torch.Tensor) -> str:
        """
        Converts tensor with token ids back to text.
        Args:
            token_ids: torch tensor with token ids.

        Returns:
            text
        """
        if token_ids.dim() != 1:
            raise ValueError('ids must have dimension 1')

        text = ''
        for token in token_ids[1:]:
            token = self.inverse[token.item()]
            if token == '<END>':
                break
            text += token if self.print_nan or token != '<NAN>' else ' '

        return text
