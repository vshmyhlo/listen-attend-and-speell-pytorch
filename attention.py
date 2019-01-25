import math
import torch.nn as nn
import torch.nn.functional as F
import torch


# TODO: move to modules
class NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale = nn.Parameter(torch.Tensor(out_features, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.scale, math.sqrt(1. / in_features))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        norm = torch.norm(self.weight, dim=-1, keepdim=True)
        weight = self.scale * self.weight / norm

        return F.linear(input, weight, self.bias)


class QKVScaledDotProductAttention(nn.Module):
    def __init__(self, size):
        super().__init__()

        # TODO: use bias or norm?
        self.query = nn.Linear(size, size)
        self.key = nn.Linear(size, size)
        self.value = nn.Linear(size, size)

        # nn.init.normal_(self.query.weight, 0, math.sqrt(2.0 / (size + size)))
        # nn.init.normal_(self.key.weight, 0, math.sqrt(2.0 / (size * 2 + size)))
        # nn.init.normal_(self.value.weight, 0, math.sqrt(2.0 / (size * 2 + size)))

        nn.init.normal_(self.query.weight, 0, 0.01)
        nn.init.normal_(self.key.weight, 0, 0.01)
        nn.init.normal_(self.value.weight, 0, 0.01)

        nn.init.constant_(self.query.bias, 0)
        nn.init.constant_(self.key.bias, 0)
        nn.init.constant_(self.value.bias, 0)

    def forward(self, input, features, features_mask):
        query = self.query(input).unsqueeze(-1)
        keys = self.key(features)
        values = self.value(features)
        del input, features

        size = keys.size(2)
        assert size == query.size(1)

        scores = torch.bmm(keys, query) / math.sqrt(size)
        scores.masked_fill_(features_mask.unsqueeze(-1) == 0, float('-inf'))

        weights = scores.softmax(1)
        context = (values * weights).sum(1)

        return context, weights


class DotProductAttention(nn.Module):
    def __init__(self, scale=True):
        super().__init__()

        if scale:
            self.scale = nn.Linear(1, 1, bias=False)
        else:
            self.scale = None

    def forward(self, input, features, features_mask):
        query = input.unsqueeze(-1)
        keys = features
        values = features
        del input, features

        size = keys.size(2)
        assert size == query.size(1)

        scores = torch.bmm(keys, query)
        if self.scale is not None:
            scores = self.scale(scores)
        scores.masked_fill_(features_mask.unsqueeze(-1) == 0, float('-inf'))

        weights = scores.softmax(1)
        context = (values * weights).sum(1)

        return context, weights


class AdditiveAttention(nn.Module):
    def __init__(self, size, normalize=True):
        super().__init__()

        self.tanh = nn.Tanh()

        if normalize:
            self.project = NormalizedLinear(size, 1, bias=False)
            self.bias = nn.Parameter(torch.Tensor(size))
            nn.init.constant_(self.bias, 0)
        else:
            self.project = nn.Linear(size, 1, bias=False)
            self.bias = None

        # TODO: init

    def forward(self, input, features, features_mask):
        query = input.unsqueeze(1)
        keys = features
        values = features
        del input, features

        size = keys.size(2)
        assert size == query.size(2)

        if self.bias is not None:
            scores = self.project(self.tanh(query + keys + self.bias))
        else:
            scores = self.project(self.tanh(query + keys))
        scores.masked_fill_(features_mask.unsqueeze(-1) == 0, float('-inf'))

        weights = scores.softmax(1)
        context = (values * weights).sum(1)

        return context, weights
