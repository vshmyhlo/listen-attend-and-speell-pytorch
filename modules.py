import torch.nn as nn
import torch.nn.functional as F


class ConvNorm1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.norm.weight, 1)
        nn.init.constant_(self.norm.bias, 0)

    def forward(self, input):
        input = self.conv(input)
        input = self.norm(input)

        return input


class ResidualBlockBasic1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_1 = ConvNorm1d(in_channels, out_channels, 3, padding=1)
        self.conv_2 = ConvNorm1d(out_channels, out_channels, 3, padding=1)

    def forward(self, input):
        residual = input

        input = self.conv_1(input)
        input = F.relu(input, inplace=True)
        input = self.conv_2(input)
        input += residual
        input = F.relu(input, inplace=True)

        return input
