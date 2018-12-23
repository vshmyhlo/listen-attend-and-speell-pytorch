import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import modules
import torchvision


# todo mel scale ref


class PBLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)  # TODO:

    def forward(self, input):
        if input.size(1) % 2 == 1:
            input = F.pad(input, (0, 0, 0, 1, 0, 0), mode='constant', value=0.)

        input = input.contiguous().view(input.size(0), input.size(1) // 2, input.size(2) * 2)
        input, hidden = self.rnn(input)

        return input, hidden


class PyramidRNNEncoder(nn.Module):
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


# TODO: more filters
class ConvRNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            modules.ConvNorm1d(80, 64, 7, stride=2, padding=3),
            nn.MaxPool1d(3, 2),
            modules.ResidualBlockBasic1d(64, 64),
            modules.ResidualBlockBasic1d(64, 64),

            modules.ConvNorm1d(64, 128, 3, stride=2, padding=1),
            modules.ResidualBlockBasic1d(128, 128),
            modules.ResidualBlockBasic1d(128, 128),

            modules.ConvNorm1d(128, 256, 3, stride=2, padding=1),
            modules.ResidualBlockBasic1d(256, 256),
            modules.ResidualBlockBasic1d(256, 256))

        self.rnn = nn.LSTM(256, 256, num_layers=1, batch_first=True, bidirectional=True)  # TODO: num layers
        # self.rnn = nn.GRU(256, 256, num_layers=1, batch_first=True, bidirectional=True)  # TODO: num layers

    def forward(self, input):
        input = input.permute(0, 2, 1)
        input = self.conv(input)
        input = input.permute(0, 2, 1)
        input, _ = self.rnn(input)

        return input


# TODO: check this is valid
class Attention(nn.Module):
    def __init__(self, size):
        super().__init__()

        # TODO: use bias or norm?
        self.query = nn.Linear(size, size)
        self.key = nn.Linear(size * 2, size)
        self.value = nn.Linear(size * 2, size)

        nn.init.normal_(self.query.weight, 0, 0.01)
        nn.init.normal_(self.key.weight, 0, 0.01)
        nn.init.normal_(self.value.weight, 0, 0.01)

        nn.init.constant_(self.query.bias, 0)
        nn.init.constant_(self.key.bias, 0)
        nn.init.constant_(self.value.bias, 0)

    def forward(self, input, features):
        query = self.query(input).unsqueeze(-1)
        keys = self.key(features)
        values = self.value(features)

        size = keys.size(2)
        assert size == query.size(1)
        scores = torch.bmm(keys, query) / math.sqrt(size)

        weights = scores.softmax(1)
        context = (values * weights).sum(1)

        return context, weights


class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # TODO:
        size = 256

        self.embedding = nn.Embedding(vocab_size, size, padding_idx=0)
        # TODO: cell type
        self.rnn = nn.LSTMCell(size * 2, size)
        # self.rnn = nn.GRUCell(size * 2, size)
        self.attention = Attention(size)
        self.output = nn.Linear(size * 2, vocab_size)

    def forward(self, input, features):
        embeddings = self.embedding(input)

        # TODO: better init
        context = torch.zeros(embeddings.size(0), embeddings.size(2)).to(embeddings.device)
        # context = last_hidden.sum(0)
        # context, _ = self.attention(torch.zeros(input.size(0), self.rnn.hidden_size).to(input.device), features)

        hidden = None
        outputs = []
        weights = []

        for t in range(embeddings.size(1)):
            input = torch.cat([embeddings[:, t, :], context], 1)
            hidden = self.rnn(input, hidden)
            output, _ = hidden
            # output = hidden
            context, weight = self.attention(output, features)
            output = torch.cat([output, context], 1)
            output = self.output(output)
            outputs.append(output)
            weights.append(weight.squeeze(-1))

        outputs = torch.stack(outputs, 1)
        weights = torch.stack(weights, 1)

        return outputs, weights


class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # self.encoder = PyramidRNNEncoder()
        self.encoder = ConvRNNEncoder()
        self.decoder = Decoder(vocab_size)

    def forward(self, spectras, seqs):
        features = self.encoder(spectras)
        logits, weights = self.decoder(seqs, features)

        return logits, weights
