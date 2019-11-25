import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size

    def forward(self, query, key):
        _, _, t = query.size()

        mask = torch.ones(1, 1, t, dtype=torch.bool, device=query.device)

        pad = (self.kernel_size // 2, self.kernel_size // 2)
        key = F.pad(key, pad, mode='constant', value=0.)
        mask = F.pad(mask, pad, mode='constant', value=False)

        keys = []
        masks = []
        for i in range(self.kernel_size):
            keys.append(key[:, :, i:i + t])
            masks.append(mask[:, :, i:i + t])
        key = torch.stack(keys, 2)
        mask = torch.cat(masks, 1)
        del keys, masks

        query = query.unsqueeze(2)
        dot = (key * query).sum(1)
        dot = dot.masked_fill_(~mask, float('-inf'))
        weights = dot.softmax(1)

        return weights


query = torch.zeros((8, 32, 100))
key = torch.zeros((8, 32, 100))

m = M(5)

out = m(query, key)
print(out.shape)
