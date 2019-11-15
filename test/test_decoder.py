import torch

from decoder import AttentionDecoder, AttentionRNNDecoder


def test_attention_rnn__decoder():
    model = AttentionRNNDecoder(8, 100)

    input = torch.zeros(1, 5, dtype=torch.long)
    features = torch.zeros(1, 7, 8)
    input_mask = None
    features_mask = torch.ones(features.size()[:2], dtype=torch.bool)

    output, hidden, etc = model(input, features, input_mask, features_mask)

    assert output.size() == (1, 5, 100)
    assert hidden.size() == (1, 1, 8)
    assert len(etc['weights']) == 1
    assert etc['weights']['enc'].size() == (1, 1, 5, 7)

    output, hidden, etc = model.infer(features, features_mask, 1, 2, 5)

    assert output.size() == (1, 5, 100)
    assert hidden.size() == (1, 1, 8)
    assert len(etc['weights']) == 1
    assert etc['weights']['enc'].size() == (1, 1, 5, 7)


def test_attention_decoder():
    model = AttentionDecoder(8, 100)

    input = torch.zeros(1, 5, dtype=torch.long)
    features = torch.zeros(1, 7, 8)
    input_mask = torch.ones_like(input, dtype=torch.bool)
    features_mask = torch.ones(features.size()[:2], dtype=torch.bool)

    output, hidden, etc = model(input, features, input_mask, features_mask)

    assert output.size() == (1, 5, 100)
    assert hidden is None
    assert len(etc['weights']) == 2
    assert etc['weights']['self'].size() == (1, 1, 5, 5)
    assert etc['weights']['enc'].size() == (1, 1, 5, 7)

    # output, hidden, etc = model.infer(features, features_mask, 1, 2, 10)
