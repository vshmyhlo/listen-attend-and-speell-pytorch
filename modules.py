import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def forward(self, input):
        _, t, c = input.size()

        pos = torch.arange(t, dtype=input.dtype, device=input.device).unsqueeze(1)
        i = torch.arange(c, dtype=input.dtype, device=input.device).unsqueeze(0)
        enc = pos / 10000**(2 * i / c)
        enc = torch.cat([
            torch.sin(enc[:, 0::2]),
            torch.cos(enc[:, 1::2]),
        ], 1)
        enc = enc.unsqueeze(0)

        input = input + enc

        return input


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


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv_norm_1 = ConvNorm2d(in_channels, out_channels // 4, 1)
        self.conv_norm_2 = ConvNorm2d(out_channels // 4, out_channels // 4, 3, stride=stride, padding=1)
        self.conv_norm_3 = ConvNorm2d(out_channels // 4, out_channels, 1)
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


class Spectrogram(nn.Module):
    def __init__(self, rate):
        super().__init__()

        self.n_fft = round(0.025 * rate)
        self.hop_length = round(0.01 * rate)

        filters = librosa.filters.mel(rate, self.n_fft)
        self.mel = nn.Conv1d(filters.shape[1], filters.shape[0], 1, bias=False)
        self.mel.weight.data.copy_(filters_to_tensor(filters))
        self.mel.weight.requires_grad = False

        self.norm = nn.BatchNorm2d(1)

    def forward(self, input):
        input = torch.stft(input, n_fft=self.n_fft, hop_length=self.hop_length)
        input = torch.norm(input, 2, -1)**2  # TODO:

        input = torch.stack([self.mel(input)], 1)
        amin = torch.tensor(1e-10).to(input.device)
        input = 10.0 * torch.log10(torch.max(amin, input))

        input = self.norm(input)

        return input


def filters_to_tensor(filters):
    filters = filters.reshape((*filters.shape, 1))
    filters = torch.tensor(filters).float()

    return filters


def downsample_mask(input, size):
    assert input.dim() == 2
    assert input.dtype == torch.bool

    input = input.unsqueeze(1).float()
    input = F.interpolate(input, size=size, mode='nearest')
    input = input.squeeze(1).bool()

    return input
