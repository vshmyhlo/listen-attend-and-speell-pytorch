from collections import Counter

from tqdm import tqdm

CHAR_VOCAB = [
    ' ', "'", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y', 'Z']


class CharVocab(object):
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
        return ''.join(self.id2sym[id] for id in ids)


class WordVocab(object):
    def __init__(self, syms):
        counter = Counter()
        for s in tqdm(syms, desc='building vocab'):
            counter.update(s.split())
        vocab = [key for key, _ in counter.most_common(25000)]
        print('vocab size: {}'.format(len(vocab)))
        vocab = ['<p>', '<unk>', '<s>', '</s>'] + list(vocab)

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
        return [self.sym2id[sym] if sym in self.sym2id else self.sym2id['<unk>'] for sym in syms.split()]

    def decode(self, ids):
        return ' '.join(self.id2sym[id] for id in ids)
