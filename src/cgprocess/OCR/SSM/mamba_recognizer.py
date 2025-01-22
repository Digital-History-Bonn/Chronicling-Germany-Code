"""Module for mamba based OCR model."""
from typing import List, Optional

import torch
from torch import nn
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams

from src.cgprocess.OCR.shared.tokenizer import Tokenizer


def create_empty_dict(length: int):
    return {None for _ in range(length)}


class Recognizer(nn.Module):
    """Implements OCR model composed of a visual encoder and a sequence decoder."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder = Encoder(cfg["encoder"])
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["encoder"]["block"]["dim"])
        self.decoder = Decoder(cfg["decoder"], self.encoder.expansion_factor)

        self.tokenizer = Tokenizer(**cfg["tokenizer"])
        self.confidence_threshold = cfg["confidence_threshold"]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass for training with the target sequence as additional input for the decoder, after
        processed through the embedding layer.
        Args:
            input: Image data with shape[B,C,H,W]
            target: token ids with shape [B,L]"""
        encoder_tokens = self.encoder(input)
        decoder_tokens = self.decoder(torch.cat((encoder_tokens, self.embedding(target)), 1))
        return decoder_tokens  # type:ignore

    def generate(self, encoder_tokens: torch.Tensor, batch_size: int):
        """Generate OCR output at inference time. This is done
        in an autoregressive way, passing each output back to the model, and ends with an end token.
        Args:
            encoder_tokens: encoder processed tokens with shape [B,C,L]
        """
        # TODO: positional encodings?
        start_token = self.tokenizer.single_token('<START>')
        end_token = self.tokenizer.single_token('<END>')
        nan_token = self.tokenizer.single_token('<NAN>')
        result_tokens = [[start_token]] * batch_size
        start_token = self.embedding(start_token)
        start_list = []
        for i in range(batch_size):
            start_list.append(start_token.clone())
        input_batch = torch.stack(start_list)

        self.decoder.allocate_inference_cache(batch_size, 0)
        self.decoder(encoder_tokens)
        while True:
            result_batch = torch.nn.functional.softmax(self.decoder(input_batch), dim=1)

            max_tensor, argmax = torch.max(result_batch, dim=1)
            argmax = argmax.type(torch.uint8)
            argmax[max_tensor < self.confidence_threshold] = nan_token
            result_tensor = argmax.detach().cpu()  # type: ignore

            for i, result in enumerate(result_tensor.tolist()):
                result_tokens[i] += [result]
            input_batch = self.embedding(result_tensor)

            if all(result[-1] == end_token for result in result_tokens):
                break
        return [self.tokenizer.to_text(torch.tensor(result)) for result in result_tokens]


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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Executes encoder layers
        Args:
            image: Image with shape [B,C,H,W]
        Returns:
            tokens: tokens with shape [B,C,L]
        """
        image = self.conv1(image)
        image = self.conv2(image)
        tokens = image.flatten(1, 2)
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

        self.layers = nn.ModuleList()
        expansion_factor = encoder_expansion
        for layer in layers:
            self.layers.append(SSMLayer(layer, expansion_factor, cfg["layers"]["downscale"], cfg["block"]))
            expansion_factor *= 2
        self.bn = torch.nn.BatchNorm1d(cfg["block"]["dim"] * expansion_factor)
        self.lm_head = nn.Linear(cfg["block"]["dim"], cfg["vocab_size"], bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Executes decoder layers
        Args:
            tokens: tokens with shape [B,C,L]
        Returns:
            tokens: tokens with shape [B,C,L]
        """

        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.bn(tokens)
        tokens = self.lm_head(tokens)
        return tokens

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int) -> None:
        """
        Returns the inference parameters for all layers. This allows for efficient inference.
        """
        param_example = next(iter(self.parameters()))
        dtype = param_example.dtype
        for layer in self.layers:
            layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)


class SSMLayer(nn.Module):
    """Implements a Layer consisting of multiple mamba blocks and an initial downscaling convolution ."""

    def __init__(self, num_blocks: int, layer_factor: int, downscale: bool, block_config: dict):
        """Creates multiple mamba blocks and an initial downscaling convolution."""
        super().__init__()
        channels = block_config["dim"] * layer_factor
        if downscale:
            self.conv = nn.Conv1d(
                channels,
                channels * 2,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            channels *= 2
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(SSMBlock(block_config, channels))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Executes SSM Layer consisting out of multiple mamba blocks and an initial downscaling convolution.
        Args:
            tokens: tokens with shape [B,C,L]
        Returns:
            tokens: tokens with shape [B,C,L]
        """

        tokens = self.conv(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype) -> None:
        """
        Returns the inference parameters for all blocks. This allows for efficient inference.
        """
        for block in self.blocks:
            block.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)


class SSMBlock(nn.Module):
    """Implements a Layer consisting of multiple mamba blocks and an initial downscaling convolution."""

    def __init__(self, cfg: dict, channels: int):
        """Creates a mamba block wrapped with batch norm and a residual connection, followed by a fully
        connected layer."""
        super().__init__()

        self.ssm = (Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=channels,  # Model dimension d_model
            d_state=cfg["state"],  # SSM state expansion factor, typically 64 or 128
            d_conv=cfg["conv_width"],  # Local convolution width
            expand=cfg["expand"],  # Block expansion factor
            layer_idx=0  # default id for accessing inference cache.
        ))
        self.norm = torch.nn.BatchNorm1d(channels)

        self.feed_forward = self.FeedForward(channels, channels * cfg["expand"])
        self.inference_params: Optional[InferenceParams] = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Executes SSM Block consisting out of multiple mamba blocks and an initial downscaling convolution.
        Args:
            tokens: tokens with shape [B,C,L]
        Returns:
            tokens: tokens with shape [B,C,L]
        """

        residual = tokens.clone()

        tokens = torch.permute(tokens, (0, 2, 1))  # mamba block needs shape of [B,L,C]
        tokens = self.ssm(tokens, inference_params=self.inference_params)
        tokens = torch.permute(tokens, (0, 2, 1))

        tokens = self.norm(tokens + residual)
        tokens = self.feed_forward(tokens)
        return tokens

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype) -> None:
        """
        Returns the ssm and conv state of the mamba block. This allows for efficient inference.
        """
        inference_cache = {0: self.ssm.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)}
        self.inference_params = InferenceParams(
            max_seqlen=max_seqlen,  # this is obsolete in this implementation, but necessary for compatibility.
            max_batch_size=batch_size,
            key_value_memory_dict=inference_cache,
        )


class FeedForward(torch.nn.Module):
    """Implements feed-forward layer with one hidden layer and a residual connection with add + norm."""

    def __init__(self, model_dim: int, hidden_dim: int):
        super().__init__()
        self.linear_in = torch.nn.Conv1d(model_dim, hidden_dim, 1)
        self.linear_out = torch.nn.Conv1d(hidden_dim, model_dim, 1)

        self.norm = torch.nn.BatchNorm1d(model_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Computes feed forward layer with residual connection and norm.
        """
        residual = tokens.clone()
        result = self.linear_in(tokens)
        result = torch.nn.functional.relu(result)
        result = self.linear_out(result)
        return self.norm(result + residual)  # type:ignore
