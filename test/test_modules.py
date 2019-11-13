import torch

from modules import downsample_mask


def test_downsample_mask():
    input = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    actual = downsample_mask(input, 3)

    assert actual.size() == (6, 3)
    assert actual.dtype == torch.bool

    expected = torch.tensor([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ], dtype=torch.bool)[[0, 0, 1, 1, 2, 2]]

    assert torch.all(actual == expected)
