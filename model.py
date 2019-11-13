import itertools

import torch
import torch.distributions
import torch.nn as nn

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

        return input


class Conv2dAttentionEncoder(nn.Module):
    def __init__(self, in_features, out_features, num_conv_layers):
        super().__init__()

        self.conv = Conv2dEncoder(in_features, out_features, num_conv_layers)
        self.encoding = modules.PositionalEncoding()
        self.attention = attention.QKVDotProductAttention(out_features)

    def forward(self, input, input_mask):
        input = self.conv(input)
        input_mask = modules.downsample_mask(input_mask, input.size(2))
        input = input.permute(0, 2, 1)

        input, weights = self.attention(input, self.encoding(input), input_mask)

        etc = {
            'weights': weights
        }

        return input, etc


class AttentionRNNDecoder(nn.Module):
    def __init__(self, features, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, features, padding_idx=0)
        self.dropout = nn.Dropout(0.1)
        self.rnn = nn.GRU(features, features, batch_first=True)
        self.attention = attention.QKVDotProductAttention(features)
        self.output = nn.Sequential(
            nn.Linear(features, vocab_size))

    def forward(self, inputs, features, features_mask, hidden=None):
        inputs = self.embedding(inputs)
        inputs = self.dropout(inputs)
        inputs, hidden = self.rnn(inputs, hidden)
        context, weights = self.attention(inputs, features, features_mask)
        inputs = inputs + self.dropout(context)
        inputs = self.output(inputs)

        return inputs, weights, hidden

    def infer(self, features, features_mask, sos_id, eos_id, max_steps, hidden=None):
        inputs = torch.full((features.size(0), 1), sos_id, dtype=torch.long, device=features.device)
        finished = torch.zeros((features.size(0), 1), dtype=torch.bool, device=features.device)

        all_weights = []
        all_logits = []

        for t in range(max_steps):
            logits, weights, hidden = self(inputs, features, features_mask, hidden)
            inputs = logits.argmax(2)

            all_logits.append(logits)
            all_weights.append(weights)

            finished = finished | (inputs == eos_id)
            if torch.all(finished):
                break

        all_logits = torch.cat(all_logits, 1)
        all_weights = torch.cat(all_weights, 2)

        return all_logits, all_weights, hidden


class Model(nn.Module):
    def __init__(self, sample_rate, vocab_size):
        super().__init__()

        self.spectra = modules.Spectrogram(sample_rate)
        self.encoder = Conv2dRNNEncoder(in_features=128, out_features=256, num_conv_layers=5, num_rnn_layers=1)
        self.decoder = AttentionRNNDecoder(features=256, vocab_size=vocab_size)

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
        spectras_mask = modules.downsample_mask(sigs_mask, spectras.size(3))
        features = self.encoder(spectras, spectras_mask)
        features_mask = modules.downsample_mask(spectras_mask, features.size(1))

        logits, weights, _ = self.decoder(seqs, features, features_mask)

        etc = {
            'spectras': spectras[:32],
            'weights': weights[:32],
        }

        return logits, etc

    def infer(self, sigs, sigs_mask, **kwargs):
        spectras = self.spectra(sigs)
        spectras_mask = modules.downsample_mask(sigs_mask, spectras.size(3))
        features = self.encoder(spectras, spectras_mask)
        features_mask = modules.downsample_mask(spectras_mask, features.size(1))

        logits, weights, _ = self.decoder.infer(features, features_mask, **kwargs)

        etc = {
            'spectras': spectras[:32],
            'weights': weights[:32],
        }

        return logits, etc
