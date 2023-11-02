"""CBAM module from https://github.com/Peachypie98/CBAM"""
import torch
from torch import nn
from torch.nn import functional


class SAM(nn.Module):
    """
    SAM module applies separate max and avg pooling along the channel dimension. The 2 resulting feature maps contain
    either for each pixel the maximum value of all channels or the average value.
    These 2 channels are then combined into one by a convolution.
    from https://github.com/Peachypie98/CBAM"""

    def __init__(self, bias: bool = False) -> None:
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
        SAM forward, calculate max and avg over all channels, concat the 2 feature maps, combine through convolution and
        apply to input.
        :param tensor_x: input
        :return: SAM result
        """
        max_value = torch.max(tensor_x, 1)[0].unsqueeze(1)
        avg = torch.mean(tensor_x, 1).unsqueeze(1)
        concat = torch.cat((max_value, avg), dim=1)
        output: torch.Tensor = self.conv(concat)
        output = output * tensor_x
        return output


class CAM(nn.Module):
    """
    CAM module, applies max and avg pooling to each channel. This results in 2 vectors each containing either max or avg
    values for each channel. These vectors are then processed through a mlp with one hidden layer. The same mlp
    processes both the max and avg vector. The vectors are added upon each other and applied to the input.
    Removed second mlp. In the original paper they use one mlp, which precesses both vectors.
    Rest from https://github.com/Peachypie98/CBAM
    """

    def __init__(self, channels: int, down_scaling: int) -> None:
        """
        :param channels: number of channels
        :param down_scaling: Downscaling factor for mlp. the hidden layer will have channels//r many neurons.
        """
        super().__init__()
        self.channels = channels
        self.down_scaling = down_scaling
        self.linear = nn.Sequential(
            nn.Linear(
                in_features=self.channels,
                out_features=self.channels // self.down_scaling,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.channels // self.down_scaling,
                out_features=self.channels,
                bias=True,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        CAM forward, calculate max and avg pooling, apply mlp to both vectors, add them and apply to input.
        :param tensor_x: input
        :return: CAM result
        """
        max_value = functional.adaptive_max_pool2d(inputs, output_size=1)
        avg = functional.adaptive_avg_pool2d(inputs, output_size=1)
        batches, channels, _, _ = inputs.size()
        linear_max = self.linear(max_value.view(batches, channels)).view(batches, channels, 1, 1)
        linear_avg = self.linear(avg.view(batches, channels)).view(batches, channels, 1, 1)
        output: torch.Tensor = linear_max + linear_avg
        output = torch.sigmoid(output) * inputs
        return output


class CBAM(nn.Module):
    """
    CBAM module consists out of 2 attention mechanisms. One for spatial attention of regions in the image(sam)
    and one for channel attention (cam). Those to modules calculate and apply weights for pixels or channels.
    Removed addition from output and input at the end. Rest from https://github.com/Peachypie98/CBAM"""

    def __init__(self, channels: int, down_scaling: int) -> None:
        """
        :param channels: number of channels
        :param down_scaling: Downscaling factor for mlp. the hidden layer will have channels//r many neurons.
        """
        super().__init__()
        self.channels = channels
        self.down_scaling = down_scaling
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, down_scaling=self.down_scaling)

    def forward(self, tensor_x: torch.Tensor) -> torch.Tensor:
        """
        CBAM forward. Applies first channel attention, then spatial attention.
        :param tensor_x: input
        :return: CBAM result
        """
        output: torch.Tensor = self.cam(tensor_x)
        output = self.sam(output)
        return output
