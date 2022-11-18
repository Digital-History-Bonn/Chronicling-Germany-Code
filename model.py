import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from utils import replace_substrings

"""
Most of the code of this model is from the implementation of ResNet 
from https://github.com/pytorch/vision/blob/1aef87d01eec2c0989458387fa04baebcc86ea7b/torchvision/models/resnet.py
"""

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Block(nn.Module):
    def __init__(self, layers, planes, conv_out=False):
        super(Block, self).__init__()
        self.layers = nn.Sequential(*layers)
        self.conv_out = conv_out
        self.conv = conv1x1(planes * Bottleneck.expansion, 512) if conv_out else None

    def forward(self, x):
        x = self.layers(x)

        if self.conv_out:
            copy = self.conv(x)
        else:
            copy = x  # maybe produces trouble by backpropagation!

        return x, copy

    def __repr__(self):
        string = "Block: (\n"
        for layer in self.layers:
            string += f"\t {layer}\n"
        return string + ")"


class UpScaleBlock(nn.Module):
    def __init__(self, in_up, in_copy, out_channels):
        super(UpScaleBlock, self).__init__()
        self.upscale = nn.ConvTranspose2d(in_channels=in_up, out_channels=in_up, kernel_size=2, stride=2)
        self.conv = conv3x3(in_copy + in_up, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up, copy):
        """
        forward path of tha Upscaling Block
        :param up: previous block feature map
        :param copy: copy of the corresponding contracting feature map
        :return: tensor
        """
        up = self.upscale(up)
        x = torch.concat((copy, up), 1)
        x = self.conv(x)
        return self.relu(x)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Normalize(torch.nn.Module):
    """
    Handles input normalization
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) ->  torch.Tensor:
        """
        input normalization with mean and standard deviation
        """
        normalized = ((x - means[None, :, None, None].to(x.device)) /
                      stds[None, :, None, None].to(x.device))
        return normalized


class DhSegment(nn.Module):
    def __init__(self, layers, in_channels=3, out_channel=3, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, load_resnet_weights=False):
        super(DhSegment, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.first_channels = 64

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.first_channels, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.first_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self._make_layer(self.first_channels, layers[0])
        self.block2 = self._make_layer(128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.block3 = self._make_layer(256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], conv_out=True)
        self.block4 = self._make_layer(512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], conv_out=True)

        # up-scaling
        self.up_block1 = UpScaleBlock(512, 512, 512)
        self.up_block2 = UpScaleBlock(512, 512, 256)
        self.up_block3 = UpScaleBlock(256, 256, 128)
        self.up_block4 = UpScaleBlock(128, 64, 64)
        self.up_block5 = UpScaleBlock(64, in_channels, 32)

        self.conv2 = conv1x1(32, out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

        if load_resnet_weights:
            self._load_ResNet()

        # initialize normalization
        self.register_buffer('means', torch.tensor([0]*in_channels))
        self.register_buffer('stds', torch.tensor([1]*in_channels))
        self.normalize = Normalize()

    def _make_layer(self, planes, blocks, stride=1, dilate=False, conv_out=False):
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

        layers = [Bottleneck(self.first_channels, planes, stride, downsample, self.groups,
                             self.base_width, previous_dilation, norm_layer)]
        self.first_channels = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.first_channels, planes, groups=self.groups,
                                     base_width=self.base_width, dilation=self.dilation,
                                     norm_layer=norm_layer))

        return Block(layers, planes, conv_out=conv_out)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        identity = x
        # print(f"input: x:{x.shape}, identity:{identity.shape}")

        x = self.normalize(x, self.means, self.stds)
        x = self.conv1(x)
        x = self.bn1(x)
        copy_0 = self.relu(x)
        # print(f"conv1: x:{x.shape}, copy_0:{copy_0.shape}")
        x = self.maxpool(copy_0)

        x, copy_1 = self.block1(x)
        # print(f"block1: x:{x.shape}, copy:{copy_1.shape}")
        x, copy_2 = self.block2(x)
        # print(f"block2: x:{x.shape}, copy:{copy_2.shape}")
        x, copy_3 = self.block3(x)
        # print(f"block3: x:{x.shape}, copy:{copy_3.shape}")
        _, copy_4 = self.block4(x)
        # print(f"block4: copy:{copy_4.shape}")

        # upscaling
        x = self.up_block1(copy_4, copy_3)
        # print(f"block1: x:{x.shape}, copy:{copy_2.shape}")
        x = self.up_block2(x, copy_2)
        # print(f"block2: x:{x.shape}, copy:{copy_1.shape}")
        x = self.up_block3(x, copy_1)
        # print(f"block3: x:{x.shape}, copy:{copy_0.shape}")
        x = self.up_block4(x, copy_0)
        # print(f"block4: x:{x.shape}, copy:{identity.shape}")
        x = self.up_block5(x, identity)
        # print(f"block5: x:{x.shape}")

        x = self.conv2(x)
        # print(f"out: {x.shape}")

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def save(self, path):
        if path is None or path is False:
            return
        torch.save(self.state_dict(), path)

    def load(self, path):
        if path is None or path is False:
            return
        self.load_state_dict(torch.load(path))
        self.eval()

    def _load_ResNet(self):
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',
                                              progress=True)

        replacements = {"layer1": "block1.layers",
                        "layer2": "block2.layers",
                        "layer3": "block3.layers",
                        "layer4": "block4.layers"}

        keys = list(state_dict.keys())
        for key in keys:
            new_key = replace_substrings(key, replacements)
            if key != new_key:
                state_dict[new_key] = state_dict.pop(key)

        state_dict = {key: state_dict[key] for key in self.state_dict().keys() if
                      key in state_dict.keys() and state_dict[key].size() == self.state_dict()[key].size()}

        self.load_state_dict(state_dict, strict=False)


def _dhSegment(arch, layers, pretrained, progress, **kwargs):
    net = DhSegment(layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        net.load_state_dict(state_dict)
    return net


def create_dhSegment(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dhSegment('resnet50', [3, 4, 6, 4], pretrained, progress,
                      **kwargs)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = DhSegment([3, 4, 6, 4], 1, load_resnet_weights=True)
