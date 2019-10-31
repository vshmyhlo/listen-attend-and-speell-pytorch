# Implementation of Automatic Speech Recognition inspired by [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211) paper in [PyTorch](http://pytorch.org)

* Encoder-Decoder architecture with attention
* Encoder is 2D Convolutional network over log-mel spectrogram followed by several GRU layers
* Decoder is GRU Network with Luong style attention
* Trained on LibriSpeech

### Example spectrograms
![Example spectrograms](./data/spectras.png)

### Corresponding attention matrices
![Example attention matrices](./data/weights.png)
