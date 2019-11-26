import torch

from decoder import AttentionDecoder, AttentionRNNDecoder


def test_attention_rnn_decoder():
    with torch.no_grad():
        b = 2
        model = AttentionRNNDecoder(8, 100)

        input = torch.zeros(b, 5, dtype=torch.long)
        features = torch.zeros(b, 7, 8)
        input_mask = None
        features_mask = torch.ones(features.size()[:2], dtype=torch.bool)

        output, hidden, etc = model(input, features, input_mask, features_mask)

        assert output.size() == (b, 5, 100)
        assert hidden.size() == (1, b, 8)
        assert len(etc['weights']) == 1
        assert etc['weights']['enc'].size() == (b, 1, 5, 7)

        output, hidden, etc = model.infer(features, features_mask, 1, 2, 5)

        assert output.size() == (b, 5, 100)
        assert hidden.size() == (1, b, 8)
        assert len(etc['weights']) == 1
        assert etc['weights']['enc'].size() == (b, 1, 5, 7)


def test_attention_decoder():
    with torch.no_grad():
        b = 2
        model = AttentionDecoder(8, 100)

        input = torch.zeros(b, 5, dtype=torch.long)
        features = torch.zeros(b, 7, 8)
        input_mask = torch.ones_like(input, dtype=torch.bool)
        features_mask = torch.ones(features.size()[:2], dtype=torch.bool)

        output, hidden, etc = model(input, features, input_mask, features_mask)

        assert output.size() == (b, 5, 100)
        assert hidden is None
        assert len(etc['weights']) == 2
        assert etc['weights']['self'].size() == (b, 1, 5, 5)
        assert etc['weights']['enc'].size() == (b, 1, 5, 7)

        output, hidden, etc = model.infer(features, features_mask, 1, 2, input.size(1))
        t = output.size(1)

        assert output.size() == (b, t, 100)
        assert hidden is None
        assert len(etc['weights']) == 2
        assert etc['weights']['self'].size() == (b, 1, t, t)
        assert etc['weights']['enc'].size() == (b, 1, t, 7)
