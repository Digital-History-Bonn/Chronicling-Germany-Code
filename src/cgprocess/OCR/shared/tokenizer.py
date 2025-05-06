"""Tokenizer for Transformer based OCR."""

import warnings
from typing import Dict, List

import torch
import torch.nn.functional as F

try:
    from ssr import Tokenizer  # pylint: disable=import-error
except ModuleNotFoundError:
    from abc import ABC, abstractmethod

    class Tokenizer(ABC):  # type: ignore
        """Abstract tokenizer class, enforcing pad, start, nan and end tokens."""

        def __init__(self, alphabet: Dict[str, int]):
            """
            Tokenizer for OCR.
            Args:
                alphabet(Dict[str]): alphabet for tokenization. '<PAD>', '<START>', '<NAN>', '<END>' token are
                required to have indices 0,1,2,3."""
            assert (
                alphabet["<PAD>"] == 0
                and alphabet["<START>"] == 1
                and alphabet["<NAN>"] == 2
                and alphabet["<END>"] == 3
            ), (
                "Tokenizer alphabet is required to have '<PAD>', '<START>', "
                "'<NAN>', '<END>' tokens with indices 0,1,2,3."
            )
            self.alphabet = alphabet

        @abstractmethod
        def __call__(self, text: str) -> torch.Tensor:
            """
            Tokenizes a sequence.
            Args:
                text(str): text to be tokenized.

            Returns:
                torch.Tensor: 1d tensor with token ids.
            """

        @abstractmethod
        def __len__(self) -> int:
            """Returns length of alphabet."""

        @abstractmethod
        def single_token(self, input_str: str) -> int:
            """
            Tokenizes a single character. This can include returning the index of a start, end or nan token.
            Args:
                input_str(str): text to be tokenized.

            Returns:
                int: token id.
            """

        @abstractmethod
        def single_token_to_text(self, token_id: int) -> str:
            """
             Converts single token id back to text.
             Args:
                token_id(int): token id.

            Returns:
                str: string representation of token.
            """

        @abstractmethod
        def to_text(self, token_ids: torch.Tensor) -> str:
            """
            Converts tensor with token ids back to text.
            Args:
                token_ids(torch.Tensor): torch tensor with token ids.

            Returns:
                str: Text representation of tokens.
            """


class OCRTokenizer(Tokenizer):  # type: ignore
    """Tokenizer for Transformer based OCR."""

    def __init__(
        self,
        alphabet: List[str],
        pad: bool = False,
        max_length: int = 512,
        print_nan: bool = False,
    ):
        """
        Tokenizer for Transformer based OCR.

        Args:
            alphabet: List with all tokens to use for tokenization.
            pad: pads all sequences to max length if true
            max_length: maximum length of tokenized sequences.
            print_nan: should nan token be considered when converting back to text
        """
        self.alphabet = {token: i for i, token in enumerate(alphabet)}
        super().__init__(self.alphabet)

        self.inverse = dict(enumerate(alphabet))
        self.pad = pad
        self.max_length = max_length
        self.print_special_tokens = print_nan

    def __call__(self, text: str) -> torch.Tensor:
        """
        Tokenize a sequence.
        Args:
            text: string

        Returns:
            Torch tensor with token ids.
        """
        ids = [self.__get_token(t) for t in text]
        ids.insert(0, self.alphabet["<START>"])
        ids.append(self.alphabet["<END>"])
        if self.pad:
            # pylint: disable=not-callable
            ids = F.pad(
                torch.tensor(ids),  # type: ignore
                pad=(0, self.max_length - len(ids)),
                mode="constant",
                value=self.alphabet["<PAD>"],
            )

        return torch.tensor(ids[: self.max_length], dtype=torch.long)

    def __len__(
        self,
    ) -> int:
        """returns alphabet length"""
        return len(self.alphabet)

    def single_token(self, input_str: str) -> int:
        """
        Tokenize a single character. This can include returning the index of a start, end or nan token.
        Args:
            input_str: string

        Returns:
            int with token id.
        """
        return self.__get_token(input_str)

    def single_token_to_text(self, input_id: int) -> str:
        """
        Converts single token id back to text.
        """
        return self.inverse[input_id]

    def __get_token(self, input_str: str) -> int:
        """
        Tokenize a single character.
        """
        return (
            self.alphabet[input_str]
            if input_str in self.alphabet.keys()
            else self.alphabet["<NAN>"]
        )

    def to_text(self, token_ids: torch.Tensor) -> str:
        """
        Converts tensor with token ids back to text.
        Args:
            token_ids: torch tensor with token ids.

        Returns:
            text
        """
        if token_ids.dim() != 1:
            raise ValueError("ids must have dimension 1")

        text = ""
        for token in token_ids[1:]:
            token = self.inverse[token.item()]
            if token == "<END>":
                break
            text += (
                token
                if self.print_special_tokens
                or (token not in ["<PAD>", "<START>", "<NAN>"])
                else ""
            )

        return text
