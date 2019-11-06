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

    def forward(self, input):
        input = self.conv(input)
        input = self.project(input)
        input = input.squeeze(2)
        input = input.permute(0, 2, 1)
        input, _ = self.rnn(input)

        return input


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

    # def infer(self, features, features_mask, sos_id, eos_id, max_steps, beam_size=5, hidden=None):
    #     def expand(input, axis):
    #         input = input.unsqueeze(axis)
    #
    #         repeats = [1] * input.dim()
    #         repeats[axis] = beam_size
    #         input = input.repeat(repeats)
    #
    #         return input
    #
    #     def flatten(input, axis):
    #         shape = list(input.size())
    #         new_shape = shape[:axis] + [shape[axis] * shape[axis + 1]] + shape[axis + 2:]
    #         input = input.view(new_shape)
    #
    #         return input
    #
    #     def unflatten(input, axis):
    #         shape = list(input.size())
    #         new_shape = shape[:axis] + [shape[axis] // beam_size, beam_size] + shape[axis + 1:]
    #         input = input.view(new_shape)
    #
    #         return input
    #
    #     inputs = torch.full((features.size(0), 1), sos_id, dtype=torch.long, device=features.device)
    #     # finished = torch.zeros((features.size(0), 1), dtype=torch.bool, device=features.device)
    #     scores = torch.zeros((features.size(0), 1), dtype=torch.float, device=features.device)
    #
    #     inputs = expand(inputs, 1)
    #     features = expand(features, 1)
    #     features_mask = expand(features_mask, 1)
    #     scores = expand(scores, 1)
    #
    #     features = flatten(features, 0)
    #     features_mask = flatten(features_mask, 0)
    #
    #     all_weights = []
    #     all_logits = []
    #
    #     for t in range(max_steps):
    #         # print('>' * 30)
    #         # print(inputs.shape)
    #         inputs = flatten(inputs, 0)
    #         if hidden is not None:
    #             hidden = flatten(hidden, 0)
    #             hidden = hidden.unsqueeze(0)
    #         # print(inputs.shape, features.shape, features_mask.shape)
    #
    #         logits, weights, hidden = self(inputs, features, features_mask, hidden)
    #         hidden = hidden.squeeze(0)
    #
    #         logits = unflatten(logits, 0)
    #         weights = unflatten(weights, 0)
    #         hidden = unflatten(hidden, 0)
    #         # print(logits.shape, weights.shape, hidden.shape)
    #
    #         # print('scores', scores.shape)
    #         scores = scores.unsqueeze(2) + logits.transpose(2, 3).log_softmax(2)
    #
    #         logits = logits.unsqueeze(2).repeat(1, 1, scores.size(2), 1, 1)
    #         weights = weights.unsqueeze(2).repeat(1, 1, scores.size(2), 1, 1, 1)
    #         hidden = hidden.unsqueeze(2).repeat(1, 1, scores.size(2), 1)
    #         # print(logits.shape, weights.shape, hidden.shape)
    #
    #         logits = flatten(logits, 1)
    #         weights = flatten(weights, 1)
    #         hidden = flatten(hidden, 1)
    #         scores = flatten(scores, 1)
    #         # print(logits.shape, weights.shape, hidden.shape)
    #
    #         scores, inputs = scores.topk(beam_size, 1)
    #         # print('scores', scores.shape)
    #         # fail
    #
    #         logits = logits.gather(1, inputs.unsqueeze(3).repeat(1, 1, 1, logits.size(3)))
    #         weights = weights.gather(1, inputs.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, 1, weights.size(4)))
    #         hidden = hidden.gather(1, inputs.repeat(1, 1, hidden.size(2)))
    #
    #         all_logits.append(logits)
    #         all_weights.append(weights)
    #
    #         # print(logits.shape, weights.shape, hidden.shape)
    #         if t >= 2:
    #             break
    #
    #     all_logits = torch.cat(all_logits, 2)
    #     all_weights = torch.cat(all_weights, 3)
    #
    #     print(all_logits.shape, all_weights.shape)
    #     fail
    #
    #     return all_logits, all_weights, hidden


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
        features = self.encoder(spectras)

        r = sigs.size(1) / features.size(1)
        features_mask = sigs_mask[:, ::math.floor(r)]
        features_mask = features_mask[:, :features.size(1)]

        logits, weights, _ = self.decoder(seqs, features, features_mask)

        etc = {
            'spectras': spectras[:32],
            'weights': weights[:32],
        }

        return logits, etc

    def infer(self, sigs, sigs_mask, **kwargs):
        spectras = self.spectra(sigs)
        features = self.encoder(spectras)

        r = sigs.size(1) / features.size(1)
        features_mask = sigs_mask[:, ::math.floor(r)]
        features_mask = features_mask[:, :features.size(1)]

        logits, weights, _ = self.decoder.infer(features, features_mask, **kwargs)

        etc = {
            'spectras': spectras[:32],
            'weights': weights[:32],
        }

        return logits, etc
