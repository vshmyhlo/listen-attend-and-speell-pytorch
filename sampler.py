import numpy as np
import pandas as pd
import torch.utils.data


# class BatchSampler(torch.utils.data.Sampler):
#     def __init__(self, data, shuffle=False):
#         super().__init__(data)
#
#         data = pd.DataFrame({
#             'i': range(len(data)),
#             'size': data['size'],
#         })
#         data = data.sort_values('size').copy()
#
#         cumsum = data['size'].cumsum()
#         cumsum = (cumsum - cumsum.min()) / cumsum.max()
#         bins = np.linspace(0, 1, len(data) // 48 + 1)
#
#         buckets = pd.Series([None] * len(data), data.index)
#         for i in range(bins.shape[0] - 1):
#             mask = (bins[i] <= cumsum) & (cumsum <= bins[i + 1])
#             buckets[mask] = i
#         data['bucket'] = buckets
#
#         sums = [group['size'].sum() for _, group in data.groupby('bucket')]
#         print(len(sums), np.mean(sums), np.std(sums))
#
#         self.data = data
#         self.shuffle = shuffle
#
#     def __iter__(self):
#         for _, g in self.data.groupby('bucket'):
#             yield g['i'].values
#
#     def __len__(self):
#         return len(self.data.groupby('bucket'))

class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, data, batch_size, shuffle=False, drop_last=False):
        super().__init__(data)

        data = pd.DataFrame({
            'i': range(len(data)),
            'size': data['size'],
        }).sort_values('size')

        batches = [group['i'] for _, group in data.groupby(np.arange(len(data)) // batch_size)]

        if drop_last and len(batches[-1]) != batch_size:
            batches = batches[:-1]

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
