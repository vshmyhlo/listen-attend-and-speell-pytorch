import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, attentions):
        super().__init__()

        self.attentions = nn.ModuleList(attentions)

    def forward(self, inputs, features, features_mask):
        inputs, weights = zip(*[a(inputs, features, features_mask) for a in self.attentions])

        inputs = sum(inputs)
        weights = torch.cat(weights, 1)

        return inputs, weights


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


# TODO: bias
class QKVDotProductAttention(nn.Module):
    def __init__(self, features, scale=True, bias=True):
        super().__init__()

        if scale:
            self.scale = nn.Parameter(torch.tensor(1. / math.sqrt(features)))
        else:
            self.scale = None

        # TODO: use bias or norm?
        self.query = nn.Linear(features, features, bias=bias)
        self.key = nn.Linear(features, features, bias=bias)
        self.value = nn.Linear(features, features, bias=bias)

        for l in [self.query, self.key, self.value]:
            # nn.init.normal_(l.weight, 0, math.sqrt(2.0 / (size + size)))
            nn.init.normal_(l.weight, 0, 0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, inputs, features, features_mask):
        query = self.query(inputs)
        keys = self.key(features)
        values = self.value(features)
        del inputs, features

        assert query.size(2) == keys.size(2)
        scores = torch.bmm(query, keys.transpose(1, 2))

        if self.scale is not None:
            scores = scores * self.scale
        if features_mask is not None:
            scores = apply_attention_mask(scores, features_mask)

        values = values.unsqueeze(1)
        scores = scores.unsqueeze(3)

        weights = scores.softmax(2)
        context = (values * weights).sum(2)

        weights = weights.squeeze(3).unsqueeze(1)

        return context, weights


class DotProductAttention(nn.Module):
    def __init__(self, scale=True):
        super().__init__()

        if scale:
            self.scale = nn.Linear(1, 1, bias=False)

            nn.init.constant_(self.scale.weight, 1.)
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
        if features_mask is not None:
            scores = apply_attention_mask(scores, features_mask)

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
        scores = apply_attention_mask(scores, features_mask)

        weights = scores.softmax(1)
        context = (values * weights).sum(1)

        return context, weights


def apply_attention_mask(input, mask):
    assert input.dim() == mask.dim(), \
        'invalid mask shape {} for input of shape {}'.format(tuple(mask.size()), tuple(input.size()))
    return input.masked_fill_(~mask, float('-inf'))


def build_subseq_attention_mask(size, device):
    shape = (1, size, size)
    subseq_attention_mask = torch.tril(torch.ones(shape, device=device, dtype=torch.bool))

    return subseq_attention_mask
