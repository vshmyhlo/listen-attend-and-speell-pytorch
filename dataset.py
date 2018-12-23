import librosa
import time
import soundfile
import numpy as np
import torch.utils.data
import os

VOCAB = [
    ' ', "'", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y', 'Z']

# TODO: use larger batch
# TODO: specific to spectra
MEAN, STD = -40.6916, 27.8401


class Vocab(object):
    def __init__(self, vocab):
        vocab = ['<p>', '<unk>', '<s>', '</s>'] + vocab

        self.sym2id = {sym: id for id, sym in enumerate(vocab)}
        self.id2sym = {id: sym for id, sym in enumerate(vocab)}

    def __len__(self):
        return len(self.sym2id)

    @property
    def sos_id(self):
        return self.sym2id['<s>']

    @property
    def eos_id(self):
        return self.sym2id['</s>']

    def encode(self, syms):
        return [self.sym2id[sym] if sym in self.sym2id else self.sym2id['<unk>'] for sym in syms]

    def decode(self, ids):
        return [self.id2sym[id] for id in ids]


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, path, subset):
        self.path = path
        self.subset = subset
        self.data = self.load_data(os.path.join(path, subset))
        self.vocab = Vocab(VOCAB)

    def __getitem__(self, item):
        speaker, chapter, id, syms = self.data[item]
        path = os.path.join(self.path, self.subset, speaker, chapter, '{}.flac'.format(id))
        # sig, rate = librosa.core.load(path, sr=None)
        sig, rate = soundfile.read(path, dtype=np.float32)
        n_fft = check_and_round(0.025 / (1 / rate))  # TODO: refactor
        hop_length = check_and_round(0.01 / (1 / rate))  # TODO: refactor

        # spectra = librosa.feature.mfcc(sig, sr=rate, n_mfcc=80, n_fft=n_fft, hop_length=hop_length)
        spectra = librosa.feature.melspectrogram(sig, sr=rate, n_mels=80, n_fft=n_fft, hop_length=hop_length)
        t = time.time()
        spectra = librosa.power_to_db(spectra, ref=np.max)
        print(time.time() - t)
        spectra = (spectra - MEAN) / STD

        syms = [self.vocab.sos_id] + self.vocab.encode(syms) + [self.vocab.eos_id]
        syms = np.array(syms)

        return spectra, syms

    def __len__(self):
        return len(self.data)

    def load_data(self, path):
        data = []
        for speaker in os.listdir(path):
            for chapter in os.listdir(os.path.join(path, speaker)):
                trans = os.path.join(path, speaker, chapter, '{}-{}.trans.txt'.format(speaker, chapter))
                with open(trans) as f:
                    for sample in f.read().splitlines():
                        id, syms = sample.split(' ', 1)
                        data.append((speaker, chapter, id, list(syms)))

        return data


def check_and_round(x):
    assert x.is_integer()
    return round(x)
