import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledLinear,
    ScaledConv2d,
)
from torch.autograd import Variable

EPS = torch.finfo(torch.get_default_dtype()).eps


# Conv2D encoder
class Conv2D(nn.Module):
    """Deep convolutional encoder.
    Source: https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/dprnn.py

    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        num_channels: list of int, number of channels in each layer.
        kernel_size: list of int, kernel size in each layer.
        output_size: int, dimension of the output feature.
        dropout: float, dropout ratio. Default is 0.
    """

    def __init__(
        self,
        feature_dim,
        input_size,
        num_channels,
        kernel_size,
        output_size,
        dropout=0.1,
    ):
        super().__init__()

        self.input_embed = nn.Sequential(
            ScaledLinear(feature_dim, feature_dim * input_size),
            Rearrange("b t (c f) -> b c t f", c=input_size),
            BasicNorm(input_size, channel_dim=1),
            ActivationBalancer(
                num_channels=input_size,
                channel_dim=1,
                min_positive=0.45,
                max_positive=0.55,
            ),
        )

        # create convolutional layers according to the configuration. we want to
        # keep the H and W dimensions the same as the input, so we use padding
        # to keep the size the same. Each layer contains a convolutional layer, ReLu,
        # a normalization layer, an activation layer, and a dropout layer.
        self.conv_layers = nn.ModuleList()
        for i in range(len(num_channels)):
            in_channels = input_size if i == 0 else num_channels[i - 1]
            layer = nn.Sequential(
                ScaledConv2d(
                    in_channels,
                    num_channels[i],
                    kernel_size[i],
                    stride=1,
                    padding=(kernel_size[i] - 1) // 2,
                ),
                nn.ReLU(),
                BasicNorm(num_channels[i], channel_dim=1),
                ActivationBalancer(
                    num_channels=num_channels[i],
                    channel_dim=1,
                    min_positive=0.45,
                    max_positive=0.55,
                    max_abs=6.0,
                ),
                nn.Dropout(dropout),
            )
            self.conv_layers.append(layer)

        # create output layer
        self.out_embed = nn.Sequential(
            ScaledLinear(feature_dim * num_channels[-1], output_size),
            BasicNorm(output_size),
            ActivationBalancer(
                num_channels=output_size,
                channel_dim=-1,
                min_positive=0.45,
                max_positive=0.55,
            ),
        )

    def forward(self, input):
        # input shape: B, T, F
        B, T, F = input.shape

        output = self.input_embed(input)

        # apply convolutional layers
        for layer in self.conv_layers:
            output = layer(output)

        # reshape the output to B, T, F
        output = rearrange(output, "b c t f -> b t (c f)")
        output = self.out_embed(output)

        # Apply ReLU to the output
        output = torch.relu(output)

        return output


if __name__ == "__main__":

    model = Conv2D(
        80,
        64,
        [64, 64, 128, 128],
        [3, 3, 3, 3],
        160,
        dropout=0.1,
    )
    input = torch.randn(2, 1002, 80)
    print(sum(p.numel() for p in model.parameters()))
    print(model(input).shape)
