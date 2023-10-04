"""CBAM module from https://github.com/Peachypie98/CBAM"""
import torch
from torch import nn as nn
from torch.nn import functional as F


class SAM(nn.Module):
    """
    SAM module
    """

    def __init__(self, bias=False):
        super().__init__()
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding=3,
            dilation=1,
            bias=self.bias,
        )

    def forward(self, tensor_x: torch.Tensor) -> torch.Tensor:
        """
        SAM forward module
        :param tensor_x: input
        :return: SAM result
        """
        max_value = torch.max(tensor_x, 1)[0].unsqueeze(1)
        avg = torch.mean(tensor_x, 1).unsqueeze(1)
        concat = torch.cat((max_value, avg), dim=1)
        output = self.conv(concat)
        output = output * tensor_x
        return output


class CAM(nn.Module):
    """
    CAM module
    """

    def __init__(self, channels: int, r: int) -> None:
        """
        :param channels: number of channels
        :param r: Downscaling factor for mlp. the hidden layer will have channels//r many neurons.
        """
        super().__init__()
        self.channels = channels
        self.r = r
        self.linear_max = nn.Sequential(
            nn.Linear(
                in_features=self.channels,
                out_features=self.channels // self.r,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.channels // self.r,
                out_features=self.channels,
                bias=True,
            ),
        )
        self.linear_avg = nn.Sequential(
            nn.Linear(
                in_features=self.channels,
                out_features=self.channels // self.r,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.channels // self.r,
                out_features=self.channels,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        CAM forward module
        :param tensor_x: input
        :return: CAM result
        """
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear_max(max.view(b, c)).view(b, c, 1, 1)
        linear_avg = self.linear_avg(avg.view(b, c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = torch.sigmoid(output) * x
        return output


class CBAM(nn.Module):
    """
    CBAM module
    """

    def __init__(self, channels, r):
        """
        :param channels: number of channels
        :param r: Downscaling factor for mlp. the hidden layer will have channels//r many neurons.
        """
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        CBAM forward module
        :param tensor_x: input
        :return: CBAM result
        """
        output = self.cam(x)
        output = self.sam(output)
        return output + x
