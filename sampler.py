import numpy as np
import pandas as pd
import torch.utils.data


class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, data, batch_size, shuffle=False, drop_last=False):
        super().__init__(data)

        data = pd.DataFrame({
            'i': range(len(data)),
            'size': data['size'],
        }).sort_values('size')

        batches = [group['i'] for _, group in data.groupby(np.arange(len(data)) // batch_size)]
        batches = [b for b in batches if len(b) > 0]
        if drop_last:
            batches = [b for b in batches if len(b) == batch_size]

        self.batches = batches
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            for i in np.random.permutation(len(self.batches)):
                yield self.batches[i]
        else:
            yield from self.batches

    def __len__(self):
        return len(self.batches)
