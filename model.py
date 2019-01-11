import torch.nn as nn
import attention
import torch
import torch.nn.functional as F
import modules


# TODO: better context init


# TODO:
class PBRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.rnn = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)  # TODO:

    def forward(self, input):
        if input.size(1) % 2 == 1:
            input = F.pad(input, (0, 0, 0, 1, 0, 0), mode='constant', value=0.)

        input = input.contiguous().view(input.size(0), input.size(1) // 2, input.size(2) * 2)
        input, hidden = self.rnn(input)

        return input, hidden


class PyramidRNNEncoder(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.rnn_1 = PBRNN(256, size // 2)
        self.rnn_2 = PBRNN(size * 2, size // 2)
        self.rnn_3 = PBRNN(size * 2, size // 2)

    def forward(self, input):
        input, _ = self.rnn_1(input)
        input, _ = self.rnn_2(input)
        input, last_hidden = self.rnn_3(input)

        return input, last_hidden


class Conv1dRNNEncoder(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.conv = nn.Sequential(
            modules.ConvNorm1d(128, 64, 7, stride=2, padding=3),

            nn.MaxPool1d(3, 2),
            modules.ResidualBlockBasic1d(64, 64),
            modules.ResidualBlockBasic1d(64, 64),

            modules.ResidualBlockBasic1d(
                64, 128, stride=2, downsample=modules.ConvNorm1d(64, 128, 3, stride=2, padding=1)),
            modules.ResidualBlockBasic1d(128, 128),

            modules.ResidualBlockBasic1d(
                128, 256, stride=2, downsample=modules.ConvNorm1d(128, 256, 3, stride=2, padding=1)),
            modules.ResidualBlockBasic1d(256, 256))

        self.rnn = nn.GRU(256, size // 2, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, input):
        input = input.permute(0, 2, 1)
        input = self.conv(input)
        input = input.permute(0, 2, 1)
        input, last_hidden = self.rnn(input)

        return input, last_hidden


class DeepConv1dRNNEncoder(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.conv = nn.Sequential(
            # modules.ConvNorm1d(128, 64, 7, stride=2, padding=3),
            modules.ConvNorm1d(128, 64, 7, stride=1, padding=3),
            nn.MaxPool1d(3, 2),

            modules.ResidualBlockBasic1d(64, 64),
            modules.ResidualBlockBasic1d(64, 64),
            modules.ResidualBlockBasic1d(64, 64),

            modules.ResidualBlockBasic1d(
                64, 128, stride=2, downsample=modules.ConvNorm1d(64, 128, 3, stride=2, padding=1)),
            modules.ResidualBlockBasic1d(128, 128),
            modules.ResidualBlockBasic1d(128, 128),
            modules.ResidualBlockBasic1d(128, 128),

            modules.ResidualBlockBasic1d(
                128, 256, stride=2, downsample=modules.ConvNorm1d(128, 256, 3, stride=2, padding=1)),
            modules.ResidualBlockBasic1d(256, 256),
            modules.ResidualBlockBasic1d(256, 256),
            modules.ResidualBlockBasic1d(256, 256),
            modules.ResidualBlockBasic1d(256, 256),
            modules.ResidualBlockBasic1d(256, 256))

        self.rnn = nn.GRU(256, size // 2, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, input):
        input = input.permute(0, 2, 1)
        input = self.conv(input)
        input = input.permute(0, 2, 1)
        input, last_hidden = self.rnn(input)

        return input, last_hidden


# class CTCEncoder(nn.Module):
#     def __init__(self, size):
#         super().__init__()
#
#         self.conv = nn.Sequential(
#             modules.ConvNorm1d(128, 256, 7, stride=2, padding=3),
#             # nn.MaxPool1d(3, 2),
#             modules.ResidualBlockBasic1d(256, 256),
#             modules.ResidualBlockBasic1d(256, 256),
#             modules.ResidualBlockBasic1d(256, 256),
#             modules.ResidualBlockBasic1d(256, 256),
#             modules.ResidualBlockBasic1d(256, 256),
#             modules.ResidualBlockBasic1d(256, 256))
#
#         self.rnn = nn.GRU(256, size // 2, num_layers=3, batch_first=True, bidirectional=True)
#
#     def forward(self, input):
#         input = input.permute(0, 2, 1)
#         input = self.conv(input)
#         input = input.permute(0, 2, 1)
#         input, last_hidden = self.rnn(input)
#
#         return input, last_hidden

class CTCEncoder(nn.Module):
    class RNN(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.rnn = nn.GRU(in_channels, out_channels, batch_first=True, bidirectional=True)
            self.norm = nn.BatchNorm1d(out_channels * 2)

        def forward(self, input):
            input, _ = self.rnn(input)
            input = input.permute(0, 2, 1)
            input = self.norm(input)
            input = input.permute(0, 2, 1)

            return input

    def __init__(self, size):
        super().__init__()

        self.conv = nn.Sequential(
            modules.ConvNorm1d(128, size, 3, padding=1),
            modules.ConvNorm1d(size, size, 3, padding=1),
            modules.ConvNorm1d(size, size, 3, stride=2, padding=1))

        self.rnn = nn.Sequential(
            self.RNN(size, size // 2),
            self.RNN(size, size // 2),
            self.RNN(size, size // 2),
            self.RNN(size, size // 2),
            self.RNN(size, size // 2))

    def forward(self, input):
        input = input.permute(0, 2, 1)
        input = self.conv(input)
        input = input.permute(0, 2, 1)
        input = self.rnn(input)

        return input


class AttentionDecoder(nn.Module):
    def __init__(self, size, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, size, padding_idx=0)
        self.rnn = nn.GRUCell(size * 2, size)
        self.attention = attention.DotProductAttention()
        self.output = nn.Linear(size * 2, vocab_size)

    def forward(self, input, features, last_hidden):
        embeddings = self.embedding(input)
        context = torch.zeros(embeddings.size(0), embeddings.size(2)).to(embeddings.device)
        hidden = None

        outputs = []
        weights = []

        for t in range(embeddings.size(1)):
            input = torch.cat([embeddings[:, t, :], context], 1)
            hidden = self.rnn(input, hidden)
            output = hidden
            context, weight = self.attention(output, features)
            output = torch.cat([output, context], 1)
            output = self.output(output)
            outputs.append(output)
            weights.append(weight.squeeze(-1))

        outputs = torch.stack(outputs, 1)
        weights = torch.stack(weights, 1)

        return outputs, weights


class HybridAttentionDecoder(nn.Module):
    def __init__(self, size, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, size, padding_idx=0)
        self.rnn = nn.GRUCell(size * 2, size)
        self.attention = attention.HyrbidAttention(size)
        self.output = nn.Linear(size * 2, vocab_size)

    def forward(self, input, features, last_hidden):
        embeddings = self.embedding(input)
        context = torch.zeros(features.size(0), features.size(2)).to(features.device)
        weight = torch.zeros(features.size(0), features.size(1)).to(features.device)
        hidden = None

        outputs = []
        weights = []

        for t in range(embeddings.size(1)):
            input = torch.cat([embeddings[:, t, :], context], 1)
            hidden = self.rnn(input, hidden)
            output = hidden
            context, weight = self.attention(output, features, weight)
            output = torch.cat([output, context], 1)
            output = self.output(output)
            outputs.append(output)
            weights.append(weight)

        outputs = torch.stack(outputs, 1)
        weights = torch.stack(weights, 1)

        return outputs, weights


class DeepAttentionDecoder(nn.Module):
    def __init__(self, size, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, size, padding_idx=0)
        self.rnn_1 = nn.GRUCell(size * 2, size)
        self.rnn_2 = nn.GRUCell(size * 2, size)
        self.attention = attention.DotProductAttention()
        self.output = nn.Linear(size, vocab_size)

    def forward(self, input, features, last_hidden):
        embeddings = self.embedding(input)
        context = torch.zeros(embeddings.size(0), embeddings.size(2)).to(embeddings.device)
        hidden_1 = None
        hidden_2 = None

        outputs = []
        weights = []

        for t in range(embeddings.size(1)):
            input = torch.cat([embeddings[:, t, :], context], 1)
            hidden_1 = self.rnn_1(input, hidden_1)
            context, weight = self.attention(hidden_1, features)
            output = torch.cat([hidden_1, context], 1)
            hidden_2 = self.rnn_2(output, hidden_2)
            output = self.output(hidden_2)
            outputs.append(output)
            weights.append(weight.squeeze(-1))

        outputs = torch.stack(outputs, 1)
        weights = torch.stack(weights, 1)

        return outputs, weights


class Model(nn.Module):
    def __init__(self, size, vocab_size):
        super().__init__()

        self.encoder = DeepConv1dRNNEncoder(size)
        self.decoder = DeepAttentionDecoder(size, vocab_size)

    def forward(self, spectras, seqs):
        features, last_hidden = self.encoder(spectras)
        logits, weights = self.decoder(seqs, features, last_hidden)

        return logits, weights


class CTCModel(nn.Module):
    def __init__(self, size, vocab_size):
        super().__init__()

        self.encoder = CTCEncoder(size)
        self.logits = nn.Linear(size, vocab_size)

    def forward(self, spectras):
        features = self.encoder(spectras)
        logits = self.logits(features)

        return logits

    def compute_seq_lens(self, seq_lens):
        # TODO: pooling layers
        for m in self.encoder.modules():
            if type(m) == nn.modules.conv.Conv1d:
                seq_lens = seq_lens + 2 * m.padding[0] - m.dilation[0] * (m.kernel_size[0] - 1) - 1
                seq_lens = seq_lens / m.stride[0] + 1
            elif type(m) == nn.modules.conv.Conv2d:
                seq_lens = seq_lens + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1
                seq_lens = seq_lens / m.stride[1] + 1

        return seq_lens.long()
