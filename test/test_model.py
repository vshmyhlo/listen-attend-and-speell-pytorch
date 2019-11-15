import torch

from model import Model


def test_model():
    with torch.no_grad():
        model = Model(16000, 100)

        input = torch.zeros(1, 16000)
        seqs = torch.zeros(1, 10, dtype=torch.long)
        input_mask = torch.ones(1, 256, dtype=torch.bool)
        seqs_mask = torch.ones(1, 10, dtype=torch.bool)

        output, etc = model(input, seqs, input_mask, seqs_mask)

        assert output.size() == (1, 10, 100)
        assert len(etc['weights']) == 3
        assert etc['weights']['encoder/self'].size() == (1, 1, 4, 4)
        assert etc['weights']['decoder/self'].size() == (1, 1, 10, 10)
        assert etc['weights']['decoder/enc'].size() == (1, 1, 10, 4)
