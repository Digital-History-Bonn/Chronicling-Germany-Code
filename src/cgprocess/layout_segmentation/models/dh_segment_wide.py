"""Module for updated dh segment version with lower capacity but a higher receptive field."""

# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
from typing import Dict, Iterator, List, Union

import torch
from torch import nn
from torch.nn.parameter import Parameter

from cgprocess.layout_segmentation.models.dh_segment import (
    Block,
    Bottleneck,
    DhSegment,
    UpScaleBlock,
    conv1x1,
)

# pylint: disable=duplicate-code
logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """
    CNN Encoder Class, using the first three resnet 50 layers, while reducing the number of feature maps in the last
    two layers and increase downscaling.
    (https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """

    def __init__(self, dhsegment: DhSegment, in_channels: int, layers: List[int]):
        super().__init__()
        self._norm_layer = nn.BatchNorm2d
        self.base_width = dhsegment.base_width
        self.groups = dhsegment.groups
        self.dilation = dhsegment.dilation

        self.conv1 = dhsegment.conv1
        self.bn1 = dhsegment.bn1
        self.relu = dhsegment.relu
        self.maxpool = dhsegment.maxpool

        planes = 64

        self.block1 = dhsegment.block1
        self.block1.conv = conv1x1(planes * Bottleneck.expansion, 128)

        self.first_channels = planes * Bottleneck.expansion
        self.block2 = self.make_layer(
            planes,
            layers[1],
            stride=2,
            conv_out=True,
            out_channels=128
        )
        self.maxpool_block3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block3 = self.make_layer(
            planes,
            layers[2],
            stride=2,
            conv_out=True,
            out_channels=128
        )
        self.maxpool_block4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block4 = self.make_layer(
            planes,
            layers[3],
            stride=2,
            conv_out=True,
            out_channels=128
        )

        # initialize normalization
        # pylint: disable=duplicate-code
        self.register_buffer("means", torch.tensor([0] * in_channels))
        self.register_buffer("stds", torch.tensor([1] * in_channels))
        self.normalize = dhsegment.normalize

    def make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        conv_out: bool = False,
        out_channels: int = 512,
    ) -> nn.Module:
        """
        Creates a layer of the ResNet50 Encoder. First Block scales image down, but does not double the feature
        maps.
        :param planes: Number of channels of all blocks after the first one.
        :param blocks: Number of Bottleneck-Blocks
        :param stride: stride for convolutional-Layer
        :param dilate: dilate for convolutional-Layer
        :param conv_out: adds an extra convolutional-Layer to the output for the shortpath
        :return: Block of the ResNet Encoder
        """
        downsample = nn.Sequential(
            conv1x1(self.first_channels, planes * Bottleneck.expansion, stride),
            self._norm_layer(planes * Bottleneck.expansion),
        )
        layers = [
            Bottleneck(
                self.first_channels,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
            )
        ]
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    self.first_channels,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return Block(layers, planes, conv_out, out_channels)

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
        result, copy_2 = self.block2(result)
        result = self.maxpool_block3(result)
        result, copy_3 = self.block3(result)
        result = self.maxpool_block4(result)
        _, copy_4 = self.block4(result)

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


# pylint: disable=duplicate-code
class Decoder(nn.Module):
    """
    CNN Decoder class, corresponding to DhSegment Decoder from https://arxiv.org/abs/1804.10371
    """

    def __init__(self, dhsegment: DhSegment):
        super().__init__()
        self.up_block1 = UpScaleBlock(128, 128, 128, double_scaling=True)
        self.up_block2 = UpScaleBlock(128, 128, 128, double_scaling=True)
        self.up_block3 = UpScaleBlock(128, 128, 128)
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
        tensor_x: torch.Tensor = self.up_block1(
            encoder_results["copy_4"], encoder_results["copy_3"]
        )
        tensor_x = self.up_block2(tensor_x, encoder_results["copy_2"])
        tensor_x = self.up_block3(tensor_x, encoder_results["copy_1"])
        tensor_x = self.up_block4(tensor_x, encoder_results["copy_0"])
        tensor_x = self.up_block5(tensor_x, encoder_results["identity"])

        tensor_x = self.conv2(tensor_x)

        return tensor_x


class DhSegmentWide(nn.Module):
    """Implements DhSegment version with much less parameters and a much wider receptive field."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channel: int = 3,
        load_resnet_weights: bool = True,
    ) -> None:
        """
        :param in_channels: input image channels eg 3 for RGB
        :param out_channel: number of output classes
        :param load_resnet_weights: whether to load the resnet weights in the encoder
        """
        # pylint: disable=duplicate-code
        super().__init__()
        self.out_channel = out_channel
        layers = [3, 4, 6, 6]
        dhsegment = DhSegment(
            layers,
            in_channels=in_channels,
            out_channel=out_channel,
            load_resnet_weights=load_resnet_weights,
        )
        self.encoder = Encoder(dhsegment, in_channels, layers)
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
