"""Module for trans_unet"""
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
from os.path import join as pjoin
from typing import Dict, Tuple, Iterator, Union

import ml_collections
import numpy as np
import torch
import torch.nn as nn
from ml_collections.config_dict import ConfigDict
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

from src.news_seg.models.dh_segment import DhSegment

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

VIT_BACKBONE = "ViT-B_16"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config: ConfigDict, img_size: Tuple[int, int], in_channels=3):
        """
        :param config: config dict
        :param img_size: int tuple for image shape
        :param in_channels: channel count
        """
        super(Embeddings, self).__init__()
        self.config = config

        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        :param x_tensor: input
        :return: return positional embedding
        """
        x_tensor = self.patch_embeddings(x_tensor)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x_tensor = x_tensor.flatten(2)
        x_tensor = x_tensor.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x_tensor + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""

    def __init__(self, config):
        """
        :param config: config dict
        :param vis:
        """
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Encoder(nn.Module):
    """
    CNN Encoder Class, corresponding to the first resnet50 layers.
    """

    def __init__(self, dhsegment: DhSegment, in_channels: int):
        super(Encoder, self).__init__()
        self.conv1 = dhsegment.conv1
        self.bn1 = dhsegment.bn1
        self.relu = dhsegment.relu
        self.maxpool = dhsegment.maxpool

        self.block1 = dhsegment.block1
        self.block2 = dhsegment.block2
        self.block3 = dhsegment.block3

        # initialize normalization
        self.register_buffer("means", torch.tensor([0] * in_channels))
        self.register_buffer("stds", torch.tensor([1] * in_channels))
        self.normalize = dhsegment.normalize

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = self.normalize(inputs, self.means, self.stds)
        identity = result
        result = self.conv1(result)
        result = self.bn1(result)
        copy_0 = self.relu(result)
        result = self.maxpool(copy_0)

        result, copy_1 = self.block1(result)
        result, copy_2 = self.block2(result)
        result, copy_3 = self.block3(result)

        return {"result": result, "identity": identity, "copy_0": copy_0, "copy_1": copy_1, "copy_2": copy_2,
                "copy_3": copy_3}

    def freeze_encoder(self, requires_grad: bool = False) -> None:
        """
        Set requires grad of encoder to True or False. Freezes encoder weights
        :param requires_grad: freezes encoder weights if false else unfreezes the weights
        """

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
        super(Decoder, self).__init__()
        # up-scaling
        self.up_block1 = dhsegment.up_block1
        self.up_block2 = dhsegment.up_block2
        self.up_block3 = dhsegment.up_block3
        self.up_block4 = dhsegment.up_block4
        self.up_block5 = dhsegment.up_block5

        self.conv2 = dhsegment.conv2

    def forward(self, transformer_result: torch.Tensor, encoder_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        forward path of cnn decoder
        :param tranformer_result: transformer output, as matrix??
        :param encoder_results: contains saved values for scip connections of unet
        :return: a decoder result
        """
        B, n_patch, hidden = transformer_result.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        tensor_x = transformer_result.permute(0, 2, 1)
        tensor_x = tensor_x.contiguous().view(B, hidden, h, w)

        tensor_x = self.up_block1(tensor_x, encoder_results["copy_3"])
        tensor_x = self.up_block2(tensor_x, encoder_results["copy_2"])
        tensor_x = self.up_block3(tensor_x, encoder_results["copy_1"])
        tensor_x = self.up_block4(tensor_x, encoder_results["copy_0"])
        tensor_x = self.up_block5(tensor_x, encoder_results["identity"])

        tensor_x = self.conv2(tensor_x)

        return torch.Tensor(tensor_x)


class Transformer(nn.Module):
    """Implements Transformer"""

    def __init__(self, config: ConfigDict, img_size: Tuple[int, int]):
        """
        :param config: config dict from get_b16_config()
        :param img_size: unniform size of all input images
        """
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = TransformerEncoder(config)

    def forward(self, input_ids):
        embedding_output, _ = self.embeddings(input_ids)
        encoded, _ = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded



class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4 - self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    """Implements Trans-UNet vision transformer"""

    def __init__(self, img_size: Tuple[int, int], in_channels: int = 3, out_channel: int = 3,
                 load_backbone=False):
        """

        :param config:
        :param in_channels:
        :param out_channel:
        :param img_size:
        :param zero_head:
        """
        super().__init__()
        dhsegment = DhSegment([3, 4, 6, 1], in_channels=in_channels, out_channel=out_channel,
                              load_resnet_weights=True)
        self.encoder = Encoder(dhsegment, in_channels)
        self.config = get_b16_config()
        self.transformer = Transformer(self.config, img_size)
        if load_backbone:
            self.load_vit_backbone()
        self.decoder = Decoder(dhsegment)

    def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        :param x_tensor: input
        :return: unet result
        """
        encoder_results = self.encoder(x_tensor)
        x_tensor = self.transformer(encoder_results["results"])  # (B, n_patch, hidden)
        x_tensor = self.decoder(x_tensor, encoder_results)
        return x_tensor

    def save(self, path: Union[str, None]) -> None:
        """
        saves the model weights
        :param path: path to savepoint
        :return: None
        """
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

    def load_vit_backbone(self) -> None:
        """
        load vit backbone
        """
        self.load_from(np.load(f"models/{VIT_BACKBONE}.npz"))

    def load_from(self, weights):
        with torch.no_grad():

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

def get_b16_config() -> ConfigDict:
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16,16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16
    return config
