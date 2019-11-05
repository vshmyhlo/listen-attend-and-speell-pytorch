import torch

from dataset import SAMPLE_RATE
from model import Model
from vocab import CHAR_VOCAB, CharVocab

vocab = CharVocab(CHAR_VOCAB)
model = Model(SAMPLE_RATE, len(vocab))

sigs = torch.empty(4, 1024 * 10).normal_()
sigs_mask = torch.ones_like(sigs, dtype=torch.uint8)
sigs_mask & sigs_mask

logits, etc = model.infer(sigs, sigs_mask, sos_id=vocab.sos_id, eos_id=vocab.eos_id, max_steps=100)

print(logits.shape, logits.dtype)
print(etc['weights'].shape, etc['weights'].dtype)
