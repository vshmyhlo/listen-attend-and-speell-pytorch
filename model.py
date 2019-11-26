import itertools

import torch.nn as nn

import decoder
import encoder
import modules
from utils import MergeDict


class Model(nn.Module):
    def __init__(self, sample_rate, vocab_size):
        super().__init__()

        self.spectra = modules.Spectrogram(sample_rate)

        # self.encoder = Conv2dRNNEncoder(in_features=128, out_features=256, num_conv_layers=5, num_rnn_layers=1)
        # self.decoder = decoder.AttentionRNNDecoder(features=256, vocab_size=vocab_size)

        self.encoder = encoder.Conv2dAttentionEncoder(in_features=128, out_features=256, num_conv_layers=5)
        self.decoder = decoder.AttentionDecoder(features=256, vocab_size=vocab_size)

        for m in itertools.chain(
                self.encoder.modules(),
                self.decoder.modules()):
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, sigs, seqs, sigs_mask, seqs_mask):
        spectras = self.spectra(sigs)
        spectras_mask = modules.downsample_mask(sigs_mask, spectras.size(3))
        etc = MergeDict(spectras=spectras[:32])

        features, etc.merge['encoder'] = self.encoder(spectras, spectras_mask)
        features_mask = modules.downsample_mask(spectras_mask, features.size(1))

        logits, _, etc.merge['decoder'] = self.decoder(seqs, features, seqs_mask, features_mask)

        return logits, etc

    def infer(self, sigs, sigs_mask, **kwargs):
        spectras = self.spectra(sigs)
        spectras_mask = modules.downsample_mask(sigs_mask, spectras.size(3))
        etc = MergeDict(spectras=spectras[:32])

        features, etc.merge['encoder'] = self.encoder(spectras, spectras_mask)
        features_mask = modules.downsample_mask(spectras_mask, features.size(1))

        logits, _, etc.merge['decoder'] = self.decoder.infer(features, features_mask, **kwargs)

        return logits, etc
