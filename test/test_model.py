import torch

from model import Model


def test_model():
    with torch.no_grad():
        model = Model(16000, 100)

        input = torch.zeros(1, 16000)
        input_mask = torch.ones(1, 256, dtype=torch.bool)
        seqs = torch.zeros(1, 10, dtype=torch.long)
        output, etc = model(input, input_mask, seqs)
        assert output.size() == (1, 10, 100)
        for t in etc['weights']['encoder']:
            assert t.size() == (1, 1, 4, 4)
        for t in etc['weights']['decoder']:
            assert t.size() == (1, 1, 10, 4)
