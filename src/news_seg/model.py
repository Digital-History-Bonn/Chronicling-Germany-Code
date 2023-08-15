"""
Module contains a U-Net Model. The model is a replica of the dhSegment model from https://arxiv.org/abs/1804.10371
Most of the code of this model is from the implementation of ResNet
from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""
from typing import Iterator, List, Tuple, Union, Any

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn.parameter import Parameter
from torchvision.transforms.functional import normalize

# pylint: disable=locally-disabled, import-error
from utils import replace_substrings

# as this is code obtained from pytorch docstrings are not added

model_urls = {
    "resnet50": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
}


def conv3x3(
        in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """
    3x3 convolution with padding
    :param in_planes: number of input feature-maps
    :param out_planes: number of output feature-maps
    :param stride: stride for convolution
    :param groups: groups for convolution
    :param dilation: dilation for convolution
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    1x1 convolution
    :param in_planes: number of input feature-maps
    :param out_planes: number of output feature-maps
    :param stride: stride for convolution
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Block(nn.Module):
    """
    Encoder Block
    """

    def __init__(self, layers: List[Bottleneck], planes: int, conv_out: bool = False):
        """
        Encoder Block
        :param layers: List of layers (Bottleneck)
        :param planes: size of Bottleneck
        :param conv_out: adds extra convolutional layer before output
        """
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.conv_out = conv_out
        self.conv = conv1x1(planes * Bottleneck.expansion, 512)

    def forward(self, in_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward path of this Module
        :param in_x: input
        :return: output and copy for shortpath in UNet
        """
        out_x = self.layers(in_x)

        if self.conv_out:
            copy = self.conv(out_x)
        else:
            copy = out_x

        return out_x, copy

    def __repr__(self) -> str:
        """
        representation of the Module if model is printed
        :return: string representation
        """
        string = "Block: (\n"
        for layer in self.layers:
            string.join(f"\t {layer}\n")
        return string + ")"


class UpScaleBlock(nn.Module):
    """
    Decoder Block
    """

    def __init__(self, in_up: int, in_copy: int, out_channels: int):
        """
        Decoder Block
        :param in_up: number of input feature maps from up-scaling-path
        :param in_copy: number of input feature maps from shortpath
        :param out_channels: number of output feature maps
        """
        super().__init__()
        self.upscale = nn.ConvTranspose2d(
            in_channels=in_up, out_channels=in_up, kernel_size=2, stride=2
        )
        self.conv = conv3x3(in_copy + in_up, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, prev_up: torch.Tensor, copy: torch.Tensor) -> torch.Tensor:
        """
        forward path of tha Upscaling Block
        :param prev_up: previous block feature maps
        :param copy: copy of the corresponding contracting feature maps
        :return: output
        """
        prev_up = self.upscale(prev_up)
        feat_x = torch.concat((copy, prev_up), 1)
        feat_x = self.conv(feat_x)
        return torch.Tensor(self.relu(feat_x))


class Bottleneck(nn.Module):
    """
    Bottleneck Layer from ResNet
    """

    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Union[nn.Module, None] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1
    ):
        """
        Bottleneck Layer from ResNet
        :param inplanes: number of input feature-maps
        :param planes: size of Bottleneck
        :param stride: stride of conv3x3 Layer
        :param downsample: Convolutional Layer for downsampling if input dim not the same as output dim
        :param groups: groups for convolution
        :param base_width: base_width of Bottleneck
        :param dilation: dilation of conv3x3 Layer
        :param norm_layer: Layer for Normalization default is BatchNorm2d
        """
        super().__init__()
        norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, tensor_x: torch.Tensor) -> torch.Tensor:
        """
        forward path of Bottleneck
        :param tensor_x: input
        :return: output
        """
        identity = tensor_x

        out = self.conv1(tensor_x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(tensor_x)

        out += identity
        out = self.relu(out)

        return torch.Tensor(out)


class DhSegment(nn.Module):
    """
    DhSegment Model
    """

    def __init__(
            self,
            layers: List[int],
            in_channels: int = 3,
            out_channel: int = 3,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Tuple[bool, bool, bool] = (False, False, False),
            load_resnet_weights: bool = False
    ) -> None:
        """
        DhSegment Model
        :param layers: List with numbers of Bottleneck-Layer per Block in Encoder
        :param in_channels: Number of input-channels (default is 3 for rgb)
        :param out_channel: Number of output-channels (number of predicted classes)
        :param groups: groups for convolutional Layer
        :param width_per_group: base_width of bottleneck-Layer
        :param replace_stride_with_dilation: each element in the tuple indicates if we should replace the 2x2 stride
        with a dilated convolution instead
        :param norm_layer: Layer for Normalization default is BatchNorm2d
        :param load_resnet_weights: Loads weights form pretrained model if True
        """
        super().__init__()
        self.out_channel = out_channel

        norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.first_channels = 64

        self.dilation = 1
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            in_channels,
            self.first_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.first_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self._make_layer(self.first_channels, layers[0])
        self.block2 = self._make_layer(
            128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.block3 = self._make_layer(
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            conv_out=True,
        )
        self.block4 = self._make_layer(
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            conv_out=True,
        )

        # up-scaling
        self.up_block1 = UpScaleBlock(512, 512, 512)
        self.up_block2 = UpScaleBlock(512, 512, 256)
        self.up_block3 = UpScaleBlock(256, 256, 128)
        self.up_block4 = UpScaleBlock(128, 64, 64)
        self.up_block5 = UpScaleBlock(64, in_channels, 32)

        self.conv2 = conv1x1(32, out_channel)

        if load_resnet_weights:
            self._load_resnet()

        # initialize normalization
        self.register_buffer("means", torch.tensor([0] * in_channels))
        self.register_buffer("stds", torch.tensor([1] * in_channels))
        self.normalize = normalize

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
        freeze(self.block4.parameters())

        # unfreeze weights, which are not loaded
        requires_grad = True
        freeze(self.block3.conv.parameters())  # type: ignore
        freeze(self.block4.conv.parameters())  # type: ignore
        freeze(self.block4.layers[3].parameters())  # type: ignore

    def _make_layer(
            self,
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            conv_out: bool = False,
    ) -> nn.Module:
        """
        creates a Block of the ResNet Encoder
        :param planes: ???
        :param blocks: Number of Bottleneck-Blocks
        :param stride: stride for convolutional-Layer
        :param dilate: dilate for convolutional-Layer
        :param conv_out: adds a extra convolutional-Layer to the output for the shortpath
        :return: Block of the ResNet Encoder
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.first_channels != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.first_channels, planes * Bottleneck.expansion, stride),
                norm_layer(planes * Bottleneck.expansion),
            )

        layers = [
            Bottleneck(
                self.first_channels,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation
            )
        ]
        self.first_channels = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    self.first_channels,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation
                )
            )

        return Block(layers, planes, conv_out)

    def _forward_impl(self, tensor_x: torch.Tensor) -> torch.Tensor:
        """
        implementation of the forward path of dhSegment
        :param tensor_x: input
        :return: output
        """
        tensor_x = self.normalize(tensor_x, self.means, self.stds)
        identity = tensor_x
        tensor_x = self.conv1(tensor_x)
        tensor_x = self.bn1(tensor_x)
        copy_0 = self.relu(tensor_x)
        tensor_x = self.maxpool(copy_0)

        tensor_x, copy_1 = self.block1(tensor_x)
        tensor_x, copy_2 = self.block2(tensor_x)
        tensor_x, copy_3 = self.block3(tensor_x)
        _, copy_4 = self.block4(tensor_x)

        # upscaling
        tensor_x = self.up_block1(copy_4, copy_3)
        tensor_x = self.up_block2(tensor_x, copy_2)
        tensor_x = self.up_block3(tensor_x, copy_1)
        tensor_x = self.up_block4(tensor_x, copy_0)
        tensor_x = self.up_block5(tensor_x, identity)

        tensor_x = self.conv2(tensor_x)

        return tensor_x

    def forward(self, tensor_x: torch.Tensor) -> torch.Tensor:
        """
        forward path of dhSegment
        :param tensor_x: input
        :return: output
        """
        return self._forward_impl(tensor_x)

    def save(self, path: Union[str, None]) -> None:
        """
        saves the model weights
        :param path: path to savepoint
        :return: None
        """
        if path is None:
            return
        torch.save(self.state_dict(), path + ".pt")

    def load(self, path: Union[str, None]) -> None:
        """
        load the model weights
        :param path: path to savepoint
        :return: None
        """
        if path is None:
            return
        self.load_state_dict(torch.load(path))
        self.eval()

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Calls prediction and applies argmax to simplify to labels.
        :param image: input iamge
        :return: Tensor of labeled prediction
        """
        pred = self(image).argmax(dim=1).float().cpu()
        prediction = torch.squeeze(pred / self.out_channel)
        return prediction

    def _load_resnet(self) -> None:
        """
        loads the weights of the pretrained ResNet
        :return: None
        """
        state_dict = load_state_dict_from_url(model_urls["resnet50"], progress=True)

        replacements = {
            "layer1": "block1.layers",
            "layer2": "block2.layers",
            "layer3": "block3.layers",
            "layer4": "block4.layers",
        }

        keys = list(state_dict.keys())
        for key in keys:
            new_key = replace_substrings(key, replacements)
            if key != new_key:
                state_dict[new_key] = state_dict.pop(key)

        state_dict = {
            key: state_dict[key]
            for key in self.state_dict().keys()
            # pylint: disable=locally-disabled, unsubscriptable-object
            if key in state_dict.keys()
               and state_dict[key].size() == self.state_dict()[key].size()
        }

        self.load_state_dict(state_dict, strict=False)


def _dh_segment(arch: str, layers: List[int], pretrained: bool, progress: bool, **kwargs: Any) -> nn.Module:
    """
    create a dhSegment Model
    :param arch: Spring name of the ResNet-architecture for loading pretrained weights
    :param layers: List of Numbers of Bottleneck Blocks in ResNet-Blocks
    :param pretrained: bool if pretrained ResNet-weights should be load
    :param progress: bool if progressbar should be shown by loading
    :param kwargs: kwargs for Model
    :return:
    """
    net = DhSegment(layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        net.load_state_dict(state_dict)
    return net


def create_dh_segment(
        pretrained: bool = False, progress: bool = True, **kwargs: Any) -> nn.Module:
    """
    dhSegement Model from https://arxiv.org/abs/1804.10371
    :param pretrained: If True, returns a model pre-trained on ImageNet
    :param progress: If True, displays a progress bar of the download to stderr
    :param kwargs: kwargs of Model
    :return: dhSegment Model
    """
    return _dh_segment("resnet50", [3, 4, 6, 4], pretrained, progress, **kwargs)


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device")

    model = DhSegment([3, 4, 6, 4], 1, load_resnet_weights=True)
