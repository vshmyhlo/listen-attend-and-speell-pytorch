import torch.nn as nn
import torch
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


class ConvNorm2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.norm.weight, 1)
        nn.init.constant_(self.norm.bias, 0)

    def forward(self, input):
        input = self.conv(input)
        input = self.norm(input)

        return input


class ResidualBlockBasic1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv_1 = ConvNorm1d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.conv_2 = ConvNorm1d(out_channels, out_channels, 3, padding=1)
        self.downsample = downsample

    def forward(self, input):
        residual = input

        input = self.conv_1(input)
        input = F.relu(input, inplace=True)
        input = self.conv_2(input)

        if self.downsample is not None:
            input += self.downsample(residual)
        else:
            input += residual

        input = F.relu(input, inplace=True)

        return input


class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv_norm_1 = ConvNorm1d(in_channels, out_channels, 1)
        self.conv_norm_2 = ConvNorm1d(out_channels, out_channels, 3, stride=stride, padding=1)
        self.conv_norm_3 = ConvNorm1d(out_channels, out_channels * 4, 1)
        self.downsample = downsample

    def forward(self, input):
        residual = input

        input = self.conv_norm_1(input)
        input = F.relu(input, inplace=True)

        input = self.conv_norm_2(input)
        input = F.relu(input, inplace=True)

        input = self.conv_norm_3(input)

        if self.downsample is not None:
            input += self.downsample(residual)
        else:
            input += residual

        input = F.relu(input, inplace=True)

        return input


class ResidualBlockBasic2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv_1 = ConvNorm2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.conv_2 = ConvNorm2d(out_channels, out_channels, 3, padding=1)
        self.downsample = downsample

    def forward(self, input):
        residual = input

        input = self.conv_1(input)
        input = F.relu(input, inplace=True)
        input = self.conv_2(input)

        if self.downsample is not None:
            input += self.downsample(residual)
        else:
            input += residual

        input = F.relu(input, inplace=True)

        return input


class TimeDropout(nn.Module):
    def __init__(self, p):
        super().__init__()

        self.p = p
        # self.dist = torch.distributions.Categorical([p, 1 - p])

    def forward(self, input):
        assert input.dim() == 3

        if self.training:
            mask = torch.rand(input.size(0), input.size(1), 1).to(input.device)
            mask = (mask > self.p).float()

            # print(input.size())
            # print(mask.size())
            # print(mask.mean())
            # print(mask.mean() * (1 / (1 - self.p)))

            input = (input * mask) * (1 / (1 - self.p))

        return input
