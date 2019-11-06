import torch

from dataset import SAMPLE_RATE
from model import Model
from vocab import CHAR_VOCAB, CharVocab

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vocab = CharVocab(CHAR_VOCAB)
model = Model(SAMPLE_RATE, len(vocab)).to(device)

with torch.no_grad():
    sigs = torch.empty(4, 50_000).normal_().to(device)
    sigs_mask = torch.ones_like(sigs, dtype=torch.bool)
    sigs_mask & sigs_mask

    logits, etc = model.infer(sigs, sigs_mask, sos_id=vocab.sos_id, eos_id=vocab.eos_id, max_steps=100)

    print(logits.shape, logits.dtype)
    print(etc['weights'].shape, etc['weights'].dtype)
