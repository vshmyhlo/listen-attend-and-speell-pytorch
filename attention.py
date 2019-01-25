import math
import torch.nn as nn
import torch.nn.functional as F
import torch


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
    def forward(self, input, features, features_mask):
        query = input.unsqueeze(-1)
        keys = features
        values = features

        size = keys.size(2)
        assert size == query.size(1)
        scores = torch.bmm(keys, query)
        scores.masked_fill_(features_mask.unsqueeze(-1) == 0, float('-inf'))

        weights = scores.softmax(1)
        context = (values * weights).sum(1)

        return context, weights


# class ScaledDotProductAttention(nn.Module):
#     def forward(self, input, features, features_mask):
#         query = input.unsqueeze(-1)
#         keys = features
#         values = features
#
#         size = keys.size(2)
#         assert size == query.size(1)
#         scores = torch.bmm(keys, query) / math.sqrt(size)
#         scores.masked_fill_(features_mask.unsqueeze(-1) == 0, float('-inf'))
#
#         weights = scores.softmax(1)
#         context = (values * weights).sum(1)
#
#         return context, weights


# if scale:
#     # Scalar used in weight scaling
#     g = variable_scope.get_variable(
#         "attention_g", dtype=dtype,
#         initializer=init_ops.ones_initializer, shape=())
#     score = g * score
#   return score

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale = nn.Linear(1, 1, bias=False)

    def forward(self, input, features, features_mask):
        query = input.unsqueeze(-1)
        keys = features
        values = features

        size = keys.size(2)
        assert size == query.size(1)
        scores = torch.bmm(keys, query) / math.sqrt(size)

        print(scores.size())
        fail
        scores.masked_fill_(features_mask.unsqueeze(-1) == 0, float('-inf'))

        weights = scores.softmax(1)
        context = (values * weights).sum(1)

        return context, weights
