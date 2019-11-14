import torch

from decoder import AttentionDecoder


def test_attentioN_decoder():
    model = AttentionDecoder(8, 100)

    input = torch.zeros(1, 5, dtype=torch.long)
    features = torch.zeros(1, 7, 8)

    input_mask = torch.ones_like(input, dtype=torch.bool)
    features_mask = torch.ones(features.size()[:2], dtype=torch.bool)

    output, hidden, etc = model(input, features, input_mask, features_mask)
