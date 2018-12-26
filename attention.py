import math
import torch.nn as nn
import torch.nn.functional as F
import torch


class QKVScaledDotProductAttention(nn.Module):
    def __init__(self, size):
        super().__init__()

        # TODO: use bias or norm?
        self.query = nn.Linear(size, size)
        self.key = nn.Linear(size * 2, size)
        self.value = nn.Linear(size * 2, size)

        # nn.init.normal_(self.query.weight, 0, math.sqrt(2.0 / (size + size)))
        # nn.init.normal_(self.key.weight, 0, math.sqrt(2.0 / (size * 2 + size)))
        # nn.init.normal_(self.value.weight, 0, math.sqrt(2.0 / (size * 2 + size)))

        nn.init.normal_(self.query.weight, 0, 0.01)
        nn.init.normal_(self.key.weight, 0, 0.01)
        nn.init.normal_(self.value.weight, 0, 0.01)

        nn.init.constant_(self.query.bias, 0)
        nn.init.constant_(self.key.bias, 0)
        nn.init.constant_(self.value.bias, 0)

    def forward(self, input, features):
        query = self.query(input).unsqueeze(-1)
        keys = self.key(features)
        values = self.value(features)

        size = keys.size(2)
        assert size == query.size(1)
        scores = torch.bmm(keys, query) / math.sqrt(size)

        weights = scores.softmax(1)
        context = (values * weights).sum(1)

        return context, weights


class DotProductAttention(nn.Module):
    def forward(self, input, features):
        query = input.unsqueeze(-1)
        keys = features
        values = features

        size = keys.size(2)
        assert size == query.size(1)
        scores = torch.bmm(keys, query)

        weights = scores.softmax(1)
        context = (values * weights).sum(1)

        return context, weights


class ScaledDotProductAttention(nn.Module):
    def forward(self, input, features):
        query = input.unsqueeze(-1)
        keys = features
        values = features

        size = keys.size(2)
        assert size == query.size(1)
        scores = torch.bmm(keys, query) / math.sqrt(size)

        weights = scores.softmax(1)
        context = (values * weights).sum(1)

        return context, weights


class HyrbidAttention(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.input = nn.Linear(size, size // 2, bias=False)
        self.features = nn.Linear(size, size // 2, bias=False)
        self.weights = nn.Conv1d(1, size // 2, 11, padding=(11 - 1) // 2)
        self.scores = nn.Linear(size // 2, 1)

    def forward(self, input, features, weights):
        encoder_features = features
        input = self.input(input.unsqueeze(1))
        features = self.input(features)
        weights = self.weights(weights.unsqueeze(1)).permute(0, 2, 1)

        scores = input + features + weights
        scores = F.tanh(scores)
        scores = self.scores(scores)

        # weights = scores.softmax(1)
        weights = scores.sigmoid()
        weights = weights / weights.sum(1, keepdim=True)
        context = (encoder_features * weights).sum(1)

        return context, weights.squeeze(-1)
