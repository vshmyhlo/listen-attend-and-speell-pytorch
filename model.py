import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

import attention
import modules


# TODO: revisit
class PBRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)  # TODO: revisit

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


class Conv2dRNNEncoder(nn.Module):
    def __init__(self, in_features, out_features, num_layers):
        super().__init__()

        base = 32

        self.conv = nn.Sequential(
            modules.ConvNorm2d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            modules.ResidualBlockBasic2d(
                base, base * 2, stride=2,
                downsample=modules.ConvNorm2d(base, base * 2, 3, stride=2, padding=1)),
            modules.ResidualBlockBasic2d(
                base * 2, base * 4, stride=2,
                downsample=modules.ConvNorm2d(base * 2, base * 4, 3, stride=2, padding=1)),
            modules.ResidualBlockBasic2d(
                base * 4, base * 8, stride=2,
                downsample=modules.ConvNorm2d(base * 4, base * 8, 3, stride=2, padding=1)))

        self.project = nn.Sequential(
            modules.ConvNorm1d(base * 8 * (in_features // 2**3), out_features, 1),
            nn.ReLU(inplace=True))
        self.rnn = nn.GRU(out_features, out_features // 2, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, input):
        input = self.conv(input)
        input = input.view(input.size(0), input.size(1) * input.size(2), input.size(3))
        input = self.project(input)
        input = input.permute(0, 2, 1)
        input, _ = self.rnn(input)

        return input


class AttentionDecoder(nn.Module):
    def __init__(self, features, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, features, padding_idx=0)
        self.rnn = nn.GRUCell(features * 2, features)
        self.attention = attention.DotProductAttention()
        self.output = nn.Linear(features * 2, vocab_size)

    def forward(self, input, features, features_mask):
        embeddings = self.embedding(input)
        context = torch.zeros(embeddings.size(0), embeddings.size(2)).to(embeddings.device)
        hidden = None

        outputs = []
        weights = []

        for t in range(embeddings.size(1)):
            input = torch.cat([embeddings[:, t, :], context], 1)
            hidden = self.rnn(input, hidden)
            context, weight = self.attention(hidden, features, features_mask)
            output = torch.cat([hidden, context], 1)
            outputs.append(output)
            weights.append(weight.squeeze(-1))

        outputs = torch.stack(outputs, 1)
        weights = torch.stack(weights, 1)
        outputs = self.output(outputs)

        return outputs, weights


class DeepAttentionDecoder(nn.Module):
    def __init__(self, size, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, size, padding_idx=0)
        self.rnn_1 = nn.GRUCell(size * 2, size)
        self.rnn_2 = nn.GRUCell(size * 2, size)
        self.attention = attention.DotProductAttention()
        self.output = nn.Linear(size, vocab_size)

    def forward(self, input, features, features_mask):
        embeddings = self.embedding(input)
        context = torch.zeros(embeddings.size(0), embeddings.size(2)).to(embeddings.device)
        hidden_1 = None
        hidden_2 = None

        outputs = []
        weights = []

        for t in range(embeddings.size(1)):
            input = torch.cat([embeddings[:, t, :], context], 1)
            hidden_1 = self.rnn_1(input, hidden_1)
            context, weight = self.attention(hidden_1, features, features_mask)
            output = torch.cat([hidden_1, context], 1)
            hidden_2 = self.rnn_2(output, hidden_2)
            output = self.output(hidden_2)
            outputs.append(output)
            weights.append(weight.squeeze(-1))

        outputs = torch.stack(outputs, 1)
        weights = torch.stack(weights, 1)

        return outputs, weights


class Model(nn.Module):
    def __init__(self, sample_rate, vocab_size):
        super().__init__()

        features = 256

        self.spectra = modules.Spectrogram(sample_rate)
        self.encoder = Conv2dRNNEncoder(128, features, num_layers=1)
        self.decoder = AttentionDecoder(features, vocab_size)

        for m in itertools.chain(
                self.encoder.modules(),
                self.decoder.modules()):
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, sigs, sigs_mask, seqs):
        spectras = self.spectra(sigs)
        features = self.encoder(spectras)

        features_mask = None
        logits, weights = self.decoder(seqs, features, features_mask)

        etc = {
            'spectras': spectras[:32],
            'weights': weights.unsqueeze(1)[:32],
        }

        return logits, etc
