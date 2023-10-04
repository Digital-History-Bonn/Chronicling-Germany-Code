"""Module for trans_unet"""
# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
from typing import Dict, Iterator, Union

import torch
from torch import nn
from torch.nn.parameter import Parameter

from src.news_seg.models.cbam import CBAM
from src.news_seg.models.dh_segment import DhSegment

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """
    CNN Encoder Class, corresponding to the first resnet50 layers.
    """

    def __init__(self, dhsegment: DhSegment, in_channels: int):
        super().__init__()
        self.conv1 = dhsegment.conv1
        self.bn1 = dhsegment.bn1
        self.relu = dhsegment.relu
        self.maxpool = dhsegment.maxpool

        self.block1 = dhsegment.block1
        self.cbam1 = CBAM(256, 2)
        self.block2 = dhsegment.block2
        self.cbam2 = CBAM(512, 2)
        self.block3 = dhsegment.block3
        self.cbam3 = CBAM(512, 2)
        self.block4 = dhsegment.block4
        self.cbam4 = CBAM(512, 2)

        # initialize normalization
        # pylint: disable=duplicate-code
        self.register_buffer("means", torch.tensor([0] * in_channels))
        self.register_buffer("stds", torch.tensor([1] * in_channels))
        self.normalize = dhsegment.normalize

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encoder forward
        :param inputs: input tensor
        :return: dictionary with result and scip-connections
        """
        result = self.normalize(inputs, self.means, self.stds)
        identity = result
        result = self.conv1(result)
        result = self.bn1(result)
        copy_0 = self.relu(result)
        result = self.maxpool(copy_0)

        result, copy_1 = self.block1(result)
        copy_1 = self.cbam1(copy_1)
        result, copy_2 = self.block2(result)
        copy_2 = self.cbam2(copy_2)
        result, copy_3 = self.block3(result)
        copy_3 = self.cbam3(copy_3)
        _, copy_4 = self.block4(result)
        copy_4 = self.cbam4(copy_4)

        return {
            "identity": identity,
            "copy_0": copy_0,
            "copy_1": copy_1,
            "copy_2": copy_2,
            "copy_3": copy_3,
            "copy_4": copy_4,
        }

    def freeze_encoder(self, requires_grad: bool = False) -> None:
        """
        Set requires grad of encoder to True or False. Freezes encoder weights
        :param requires_grad: freezes encoder weights if false else unfreezes the weights
        """

        # noinspection DuplicatedCode
        # pylint: disable=duplicate-code
        def freeze(params: Iterator[Parameter]) -> None:
            for param in params:
                param.requires_grad_(requires_grad)

        freeze(self.conv1.parameters())
        freeze(self.bn1.parameters())
        freeze(self.block1.parameters())
        freeze(self.block2.parameters())
        freeze(self.block3.parameters())

        # unfreeze weights, which are not loaded
        requires_grad = True
        freeze(self.block3.conv.parameters())  # type: ignore


class Decoder(nn.Module):
    """
    CNN Decoder class, corresponding to DhSegment Decoder
    """

    def __init__(self, dhsegment: DhSegment):
        super().__init__()
        self.up_block1 = dhsegment.up_block1
        self.up_block2 = dhsegment.up_block2
        self.up_block3 = dhsegment.up_block3
        self.up_block4 = dhsegment.up_block4
        self.up_block5 = dhsegment.up_block5

        self.conv2 = dhsegment.conv2

    def forward(self, encoder_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        forward path of cnn decoder
        :param transformer_result: transformer output, as matrix??
        :param encoder_results: contains saved values for scip connections of unet
        :return: a decoder result
        """
        # pylint: disable=duplicate-code
        tensor_x: torch.Tensor = self.up_block1(encoder_results["copy_4"], encoder_results["copy_3"])
        tensor_x = self.up_block2(tensor_x, encoder_results["copy_2"])
        tensor_x = self.up_block3(tensor_x, encoder_results["copy_1"])
        tensor_x = self.up_block4(tensor_x, encoder_results["copy_0"])
        tensor_x = self.up_block5(tensor_x, encoder_results["identity"])

        tensor_x = self.conv2(tensor_x)

        return tensor_x


class DhSegmentCBAM(nn.Module):
    """Implements DhSegment combined with CBAM modules after encoder layers"""

    def __init__(
        self, in_channels: int = 3, out_channel: int = 3, load_resnet_weights: bool =True
    ) -> None:
        """
        :param config:
        :param in_channels:
        :param out_channel:
        :param zero_head:
        """
        super().__init__()
        dhsegment = DhSegment(
            [3, 4, 6, 1],
            in_channels=in_channels,
            out_channel=out_channel,
            load_resnet_weights=load_resnet_weights,
        )
        self.encoder = Encoder(dhsegment, in_channels)
        self.decoder = Decoder(dhsegment)

    def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        :param x_tensor: input
        :return: unet result
        """
        encoder_results = self.encoder(x_tensor)
        x_tensor = self.decoder(encoder_results)
        return x_tensor

    def freeze_encoder(self, requires_grad: bool = False) -> None:
        """
        Set requires grad of encoder to True or False. Freezes encoder weights
        :param requires_grad: freezes encoder weights if false else unfreezes the weights
        """

        self.encoder.freeze_encoder(requires_grad)

    def save(self, path: Union[str, None]) -> None:
        """
        saves the model weights
        :param path: path to savepoint
        :return: None
        """
        # pylint: disable=duplicate-code
        if path is None:
            return
        torch.save(self.state_dict(), path + ".pt")

    def load(self, path: Union[str, None], device: str) -> None:
        """
        load the model weights
        :param device: mapping device
        :param path: path to savepoint
        :return: None
        """
        if path is None:
            return
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
