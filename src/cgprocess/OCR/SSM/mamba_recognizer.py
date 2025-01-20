"""Module for mamba based OCR model."""
from typing import List

import torch
from torch import nn
from mamba_ssm import Mamba2


class Recognizer(nn.Module):
    """Implements OCR model composed of a visual encoder and a sequence decoder."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder = Encoder(cfg["encoder"])
        self.decoder = Decoder(cfg["decoder"], self.encoder.expansion_factor)

    def forward(self, encoder_tokens: torch.Tensor, decoder_tokens: torch.Tensor) -> torch.Tensor:
        """Executes encoder and decoder."""
        encoder_result = self.encoder(encoder_tokens)
        decoder_result = self.decoder(encoder_result, decoder_tokens)
        return decoder_result  # type:ignore

    def generate(self):
        """"""
        raise NotImplementedError


class Encoder(nn.Module):
    """Implements encoder with multiple layers of mamba blocks and an initial downscaling convolution."""
    def __init__(self, cfg: dict):
        """Creates ssm layers with an initial downscaling 2d convolution."""
        super().__init__()
        layers: List[int] = cfg["layers"]["num_blocks"]
        self.conv1 = nn.Conv2d(
            1,
            2,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            2,
            4,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.layers = nn.ModuleList()
        expansion_factor = 1
        for layer in layers:
            self.layers.append(SSMLayer(layer, expansion_factor, cfg["layers"]["downscale"], cfg["block"]))
            expansion_factor *= 2
        self.expansion_factor = expansion_factor

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Executes encoder layers
        """
        tokens = self.conv1(tokens)
        tokens = self.conv2(tokens)
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens


class Decoder(nn.Module):
    """Implements decoder with multiple layers of mamba blocks, a language head and embeddings autoregressive
    processing of previous outputs."""
    def __init__(self, cfg: dict, encoder_expansion: int):
        """Creates ssm layers with an initial downscaling 2d convolution."""
        super().__init__()
        layers: List[int] = cfg["layers"]["num_blocks"]

        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["block"]["dim"])

        self.layers = nn.ModuleList()
        expansion_factor = encoder_expansion
        for layer in layers:
            self.layers.append(SSMLayer(layer, expansion_factor, cfg["layers"]["downscale"], cfg["block"]))
            expansion_factor *= 2
        self.bn = torch.nn.BatchNorm1d(cfg["block"]["dim"]* expansion_factor)
        self.lm_head = nn.Linear(cfg["block"]["dim"], cfg["vocab_size"], bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Executes encoder layers
        """
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.bn(tokens)
        tokens = self.lm_head(tokens)
        return tokens


class SSMLayer(nn.Module):
    """Implements a Layer consisting of multiple mamba blocks and an initial downscaling convolution ."""
    def __init__(self, num_blocks: int, layer_factor: int, downscale: bool, block_config: dict):
        """Creates multiple mamba blocks and an initial downscaling convolution."""
        super().__init__()
        channels = block_config["dim"] * layer_factor
        self.conv = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.layers.append(Mamba2(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=channels * 2, # Model dimension d_model
                d_state=block_config["dim"],  # SSM state expansion factor, typically 64 or 128
                d_conv=block_config["conv_width"],    # Local convolution width
                expand=block_config["expand"],    # Block expansion factor
            ))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Executes SSM Layer consisting out of multiple mamba blocks and an initial downscaling convolution.
        """
        tokens = self.conv(tokens)
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens
