import torch

from encoder import Conv2dAttentionEncoder


def test_conv2d_attention_encoder():
    with torch.no_grad():
        model = Conv2dAttentionEncoder(128, 32, 5)
        input = torch.zeros(1, 1, 128, 256)
        input_mask = torch.ones(1, 256, dtype=torch.bool)
        output, etc = model(input, input_mask)
        assert output.size() == (1, 8, 32)
        assert len(etc['weights']) == 1
        assert etc['weights']['self'].size() == (1, 1, 8, 8)
       