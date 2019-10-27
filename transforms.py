import librosa
import numpy as np
import torch


class ApplyTo(object):
    def __init__(self, tos, transform):
        self.tos = tos
        self.transform = transform

    def __call__(self, input):
        input = {
            **input,
            **{to: self.transform(input[to]) for to in self.tos},
        }

        return input


class Extract(object):
    def __init__(self, fields):
        self.fields = fields

    def __call__(self, input):
        return tuple(input[k] for k in self.fields)


class LoadSignal(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, input):
        input, rate = librosa.core.load(input, sr=None, dtype=np.float32)
        assert rate == self.sample_rate

        return input


class VocabEncode(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, input):
        input = [self.vocab.sos_id] + self.vocab.encode(input) + [self.vocab.eos_id]
        input = np.array(input, dtype=np.int64)

        return input


class ToTensor(object):
    def __call__(self, input):
        input = torch.tensor(input)

        return input
