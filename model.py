import torch.nn as nn
import torch
import torch.nn.functional as F


class PBLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)  # TODO:

    def forward(self, input):
        if input.size(1) % 2 == 1:
            input = F.pad(input, (0, 0, 0, 1, 0, 0), mode='constant', value=0.)

        input = input.contiguous().view(input.size(0), input.size(1) // 2, input.size(2) * 2)
        input, hidden = self.lstm(input)

        return input, hidden


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn_1 = PBLSTM(160, 64)
        self.rnn_2 = PBLSTM(256, 128)
        self.rnn_3 = PBLSTM(512, 256)

    def forward(self, input):
        input, _ = self.rnn_1(input)
        input, _ = self.rnn_2(input)
        input, _ = self.rnn_3(input)

        return input


# TODO: shift targets


class Attention(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.query = nn.Linear(size, size)
        self.key = nn.Linear(size * 2, size)
        self.value = nn.Linear(size * 2, size)

    def forward(self, input, features):
        query = self.query(input).unsqueeze(-1)
        keys = self.key(features)
        values = self.value(features)
        scores = torch.bmm(keys, query)
        weights = scores.softmax(1)
        context = (values * weights).sum(1)

        return context


class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        size = 256

        self.embedding = nn.Embedding(vocab_size, size)
        self.rnn = nn.LSTMCell(size * 2, size)
        self.attention = Attention(size)
        self.output = nn.Linear(size * 2, vocab_size)

    def forward(self, inputs, features):
        embeddings = self.embedding(inputs)
        context = torch.zeros(embeddings.size(0), embeddings.size(2)).to(embeddings.device)
        hidden = None
        outputs = []

        for t in range(embeddings.size(1)):
            inputs = torch.cat([embeddings[:, t, :], context], 1)
            hidden = self.rnn(inputs, hidden)
            output, _ = hidden
            context = self.attention(output, features)
            output = torch.cat([output, context], 1)
            output = self.output(output)
            outputs.append(output)

        outputs = torch.stack(outputs, 1)

        return outputs


class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size)

    def forward(self, spectras, seqs):
        features = self.encoder(spectras)
        logits = self.decoder(seqs, features)

        return logits
