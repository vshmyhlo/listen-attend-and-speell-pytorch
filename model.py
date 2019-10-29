import itertools
import math

import torch
import torch.distributions
import torch.nn as nn

import attention
import modules


class Conv2dRNNEncoder(nn.Module):
    def __init__(self, in_features, out_features, num_conv_layers, num_rnn_layers, pool=True):
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

        if pool:
            self.project = nn.Sequential(
                nn.MaxPool2d((in_features // 2**num_conv_layers, 1), 1),
                modules.ConvNorm2d(base, out_features, kernel_size=1),
                nn.ReLU(inplace=True))
        else:
            raise NotImplementedError()

        self.rnn = nn.GRU(
            out_features, out_features // 2, num_layers=num_rnn_layers, batch_first=True, bidirectional=True)
        # self.norm = nn.LayerNorm(out_features)

    def forward(self, input):
        input = self.conv(input)
        input = self.project(input)
        input = input.squeeze(2)
        input = input.permute(0, 2, 1)
        input, _ = self.rnn(input)
        # input = self.norm(input)

        return input


class AttentionDecoder(nn.Module):
    def __init__(self, in_features, mid_features, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, in_features, padding_idx=0)
        self.rnn = nn.GRUCell(in_features * 2, mid_features)
        self.attention = attention.QKVDotProductAttention(mid_features)
        self.output = nn.Sequential(
            # nn.LayerNorm(mid_features * 2),
            nn.Linear(mid_features * 2, vocab_size))

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
            weights.append(weight)

        outputs = torch.stack(outputs, 1)
        weights = torch.stack(weights, 2)

        outputs = self.output(outputs)

        return outputs, weights


class SamplingAttentionDecoder(nn.Module):
    def __init__(self, in_features, mid_features, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, in_features, padding_idx=0)
        self.rnn = nn.GRUCell(in_features * 2, mid_features)
        self.attention = attention.QKVDotProductAttention(mid_features)
        self.output = nn.Sequential(
            # nn.LayerNorm(mid_features * 2),
            nn.Linear(mid_features * 2, vocab_size))

    def forward(self, input, features, features_mask):
        input_prob = self.one_hot(input, self.embedding.num_embeddings)
        context = torch.zeros(input.size(0), self.embedding.embedding_dim).to(input.device)
        hidden = None
        output = None

        outputs = []
        weights = []

        for t in range(input.size(1)):
            if output is None:
                prob = input_prob[:, t, :]
            else:
                prob = 0.5 * input_prob[:, t, :] + \
                       0.5 * output.softmax(1)

            dist = torch.distributions.Categorical(probs=prob)
            index = dist.sample()

            embedding = self.embedding(index)
            input = torch.cat([embedding, context], 1)
            hidden = self.rnn(input, hidden)
            context, weight = self.attention(hidden, features, features_mask)
            output = torch.cat([hidden, context], 1)
            output = self.output(output)
            outputs.append(output)
            weights.append(weight)

        outputs = torch.stack(outputs, 1)
        weights = torch.stack(weights, 2)

        return outputs, weights

    @staticmethod
    def one_hot(input, num_classes):
        input = torch.eye(num_classes, dtype=torch.float, device=input.device)[input]

        return input


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
        self.encoder = Conv2dRNNEncoder(in_features=128, out_features=features, num_conv_layers=5, num_rnn_layers=1)
        self.decoder = AttentionDecoder(in_features=features, mid_features=features, vocab_size=vocab_size)

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

        r = sigs.size(1) / features.size(1)
        features_mask = sigs_mask[:, ::math.floor(r)]
        features_mask = features_mask[:, :features.size(1)]

        logits, weights = self.decoder(seqs, features, features_mask)

        etc = {
            'spectras': spectras[:32],
            'weights': weights[:32],
        }

        return logits, etc
