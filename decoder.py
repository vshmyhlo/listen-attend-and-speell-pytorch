import torch
from torch import nn as nn

import attention
import modules


class AttentionRNNDecoder(nn.Module):
    def __init__(self, features, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, features, padding_idx=0)
        self.dropout = nn.Dropout(0.1)
        self.rnn = nn.GRU(features, features, batch_first=True)
        self.attention = attention.QKVDotProductAttention(features)
        self.output = nn.Sequential(
            nn.Linear(features, vocab_size))

    def forward(self, inputs, features, input_mask, features_mask, hidden=None):
        inputs = self.embedding(inputs)
        inputs = self.dropout(inputs)
        inputs, hidden = self.rnn(inputs, hidden)
        context, weights = self.attention(inputs, features, features_mask.unsqueeze(1))
        inputs = inputs + self.dropout(context)
        inputs = self.output(inputs)

        etc = {
            'weights': {
                'enc': weights,
            }
        }

        return inputs, hidden, etc

    def infer(self, features, features_mask, sos_id, eos_id, max_steps, hidden=None):
        inputs = torch.full((features.size(0), 1), sos_id, dtype=torch.long, device=features.device)
        finished = torch.zeros((features.size(0), 1), dtype=torch.bool, device=features.device)

        all_weights = []
        all_logits = []

        for t in range(max_steps):
            logits, hidden, etc = self(inputs, features, features_mask, hidden)
            inputs = logits.argmax(2)

            all_logits.append(logits)
            all_weights.append(etc['weights']['enc'])

            finished = finished | (inputs == eos_id)
            if torch.all(finished):
                break

        all_logits = torch.cat(all_logits, 1)
        all_weights = torch.cat(all_weights, 2)

        etc = {
            'weights': {
                'enc': all_weights,
            },
        }

        return all_logits, hidden, etc


class AttentionDecoder(nn.Module):
    def __init__(self, features, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, features, padding_idx=0)
        self.encoding = modules.PositionalEncoding()
        self.dropout = nn.Dropout(0.1)
        self.self_attention = attention.QKVDotProductAttention(features)
        self.attention = attention.QKVDotProductAttention(features)
        self.output = nn.Sequential(
            nn.Linear(features, vocab_size))

    def forward(self, input, features, input_mask, features_mask, hidden=None):
        input = self.embedding(input)
        input = self.encoding(input)
        input = self.dropout(input)

        subseq_attention_mask = attention.build_subseq_attention_mask(input.size(1), input.device)
        context, self_weights = self.self_attention(input, input, input_mask.unsqueeze(1) & subseq_attention_mask)
        input = input + self.dropout(context)

        context, enc_weights = self.attention(input, features, features_mask.unsqueeze(1))
        input = input + self.dropout(context)

        input = self.output(input)

        etc = {
            'weights': {
                'self': self_weights,
                'enc': enc_weights
            },
        }

        return input, hidden, etc