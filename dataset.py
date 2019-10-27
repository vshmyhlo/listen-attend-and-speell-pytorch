import os
from functools import partial
from multiprocessing import Pool

import pandas as pd
import torch.utils.data
from tqdm import tqdm

from transforms import LoadSignal

SAMPLE_RATE = 16000


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, item):
        row = self.data.iloc[item]
        sig = os.path.join(row['root'], row['speaker'], row['chapter'], '{}.flac'.format(row['id']))

        input = {
            'sig': sig,
            'syms': row['syms'],
        }

        if self.transform is not None:
            input = self.transform(input)

        return input

    def __len__(self):
        return len(self.data)


def load_speaker(speaker, path):
    loader = LoadSignal(SAMPLE_RATE)

    data = {
        'speaker': [],
        'chapter': [],
        'id': [],
        'syms': [],
        'size': [],
    }

    for chapter in os.listdir(os.path.join(path, speaker)):
        trans = os.path.join(path, speaker, chapter, '{}-{}.trans.txt'.format(speaker, chapter))
        with open(trans) as f:
            for sample in f.read().splitlines():
                id, syms = sample.split(' ', 1)

                data['speaker'].append(speaker)
                data['chapter'].append(chapter)
                data['id'].append(id)
                data['syms'].append(syms)

                sig = os.path.join(path, speaker, chapter, '{}.flac'.format(id))
                sig = loader(sig)
                data['size'].append(sig.shape[0])

    data = pd.DataFrame(data)
    data['root'] = path

    return data


def load_data(path, workers):
    with Pool(workers) as pool:
        data = pool.map(
            partial(load_speaker, path=path),
            tqdm(os.listdir(path), desc='loading {}'.format(path)))

    data = pd.concat(data)

    return data
