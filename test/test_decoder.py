import torch

from decoder import AttentionDecoder, AttentionRNNDecoder


def test_attention_rnn_decoder():
    with torch.no_grad():
        b = 2
        num_classes = 100

        model = AttentionRNNDecoder(8, num_classes)

        input = torch.zeros(b, 5, dtype=torch.long)
        features = torch.zeros(b, 7, 8)
        input_mask = None
        features_mask = torch.ones(features.size()[:2], dtype=torch.bool)

        output, hidden, etc = model(input, features, input_mask, features_mask)

        assert output.size() == (b, 5, num_classes)
        assert hidden.size() == (1, b, 8)
        assert len(etc['weights']) == 1
        assert etc['weights']['enc'].size() == (b, 1, 5, 7)

        output, hidden, etc = model.infer(features, features_mask, 1, 2, 5)

        assert output.size() == (b, 5, num_classes)
        assert hidden.size() == (1, b, 8)
        assert len(etc['weights']) == 1
        assert etc['weights']['enc'].size() == (b, 1, 5, 7)


def test_attention_decoder():
    with torch.no_grad():
        b = 2
        num_classes = 100

        model = AttentionDecoder(8, num_classes)

        input = torch.zeros(b, 5, dtype=torch.long)
        features = torch.zeros(b, 7, 8)
        input_mask = torch.ones_like(input, dtype=torch.bool)
        features_mask = torch.ones(features.size()[:2], dtype=torch.bool)

        output, hidden, etc = model(input, features, input_mask, features_mask)

        assert output.size() == (b, 5, num_classes)
        assert hidden is None
        assert len(etc['weights']) == 2
        assert etc['weights']['self'].size() == (b, 1, 5, 5)
        assert etc['weights']['enc'].size() == (b, 1, 5, 7)

        output, hidden, etc = model.infer(features, features_mask, 1, 2, input.size(1))
        t = output.size(1)

        assert output.size() == (b, t, num_classes)
        assert hidden is None
        assert len(etc['weights']) == 2
        assert etc['weights']['self'].size() == (b, 1, t, t)
        assert etc['weights']['enc'].size() == (b, 1, t, 7)

        # TODO: refactor
        input = torch.empty(b, 5, dtype=torch.long).random_(10, 20)
        features = torch.empty(b, 7, 8).normal_()
        input_mask = torch.ones_like(input, dtype=torch.bool)
        features_mask = torch.ones(features.size()[:2], dtype=torch.bool)

        model.eval()
        o1, _, _ = model(input, features, input_mask, features_mask)
        o2, _, _ = model.infer(features, features_mask, 1, 2, input.size(1), debug_input=input)

        assert torch.allclose(o1, o2, atol=1e-7)
