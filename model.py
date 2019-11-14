import itertools

import torch.nn as nn

import modules
from decoder import AttentionRNNDecoder
from encoder import Conv2dRNNEncoder


class Model(nn.Module):
    def __init__(self, sample_rate, vocab_size):
        super().__init__()

        self.spectra = modules.Spectrogram(sample_rate)

        self.encoder = Conv2dRNNEncoder(in_features=128, out_features=256, num_conv_layers=5, num_rnn_layers=1)
        # self.encoder = Conv2dAttentionEncoder(in_features=128, out_features=256, num_conv_layers=5)

        self.decoder = AttentionRNNDecoder(features=256, vocab_size=vocab_size)

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

    def forward(self, sigs, sigs_mask, seqs):
        spectras = self.spectra(sigs)
        spectras_mask = modules.downsample_mask(sigs_mask, spectras.size(3))
        features, encoder_etc = self.encoder(spectras, spectras_mask)
        features_mask = modules.downsample_mask(spectras_mask, features.size(1))

        logits, _, decoder_etc = self.decoder(seqs, features, features_mask)

        etc = {
            'spectras': spectras[:32],
            'weights': {
                'encoder': encoder_etc['weights'],
                'decoder': decoder_etc['weights'],
            }
        }

        return logits, etc

    def infer(self, sigs, sigs_mask, **kwargs):
        spectras = self.spectra(sigs)
        spectras_mask = modules.downsample_mask(sigs_mask, spectras.size(3))
        features, encoder_etc = self.encoder(spectras, spectras_mask)
        features_mask = modules.downsample_mask(spectras_mask, features.size(1))

        logits, _, decoder_etc = self.decoder.infer(features, features_mask, **kwargs)

        etc = {
            'spectras': spectras[:32],
            'weights': {
                'encoder': encoder_etc['weights'],
                'decoder': decoder_etc['weights'],
            }
        }

        return logits, etc
