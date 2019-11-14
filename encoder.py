from torch import nn as nn

import attention
import modules


class Conv2dEncoder(nn.Module):
    def __init__(self, in_features, out_features, num_conv_layers):
        super().__init__()

        base = 32
        conv = []
        for i in range(num_conv_layers):
            if i == 0:
                block = nn.Sequential(
                    modules.ConvNorm2d(1, base * 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True))
            else:
                block = modules.ResidualBlockBasic2d(
                    base, base * 2, stride=2,
                    downsample=modules.ConvNorm2d(base, base * 2, kernel_size=1, stride=2))

            conv.append(block)
            base *= 2

        self.conv = nn.Sequential(*conv)

        self.project = nn.Sequential(
            nn.MaxPool2d((in_features // 2**num_conv_layers, 1), 1),
            modules.ConvNorm2d(base, out_features, kernel_size=1),
            nn.ReLU(inplace=True))

    def forward(self, input):
        input = self.conv(input)
        input = self.project(input)
        input = input.squeeze(2)

        return input


class Conv2dRNNEncoder(nn.Module):
    def __init__(self, in_features, out_features, num_conv_layers, num_rnn_layers):
        super().__init__()

        self.conv = Conv2dEncoder(in_features, out_features, num_conv_layers)
        self.rnn = nn.GRU(
            out_features, out_features // 2, num_layers=num_rnn_layers, batch_first=True, bidirectional=True)

    def forward(self, input, input_mask):
        input = self.conv(input)
        input = input.permute(0, 2, 1)
        input, _ = self.rnn(input)

        etc = {
            'weights': [],
        }

        return input, etc


class Conv2dAttentionEncoder(nn.Module):
    def __init__(self, in_features, out_features, num_conv_layers):
        super().__init__()

        self.embedding = Conv2dEncoder(in_features, out_features, num_conv_layers)
        self.encoding = modules.PositionalEncoding()
        self.dropout = nn.Dropout(0.1)
        self.self_attention = attention.QKVDotProductAttention(out_features)

    def forward(self, input, input_mask):
        input = self.embedding(input)
        input_mask = modules.downsample_mask(input_mask, input.size(2))
        input = input.permute(0, 2, 1)

        input = self.encoding(input)
        input = self.dropout(input)

        context, weights = self.self_attention(input, input, input_mask.unsqueeze(1))
        input = input + self.dropout(context)

        etc = {
            'weights': [weights],
        }

        return input, etc
