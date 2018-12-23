from tqdm import tqdm
import torch.utils.data
from dataset import TrainEvalDataset
from train import collate_fn
import os

train_dataset = TrainEvalDataset('./data/LibriSpeech', subset='train-clean-100')
train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    # num_workers=os.cpu_count(),
    num_workers=4,
    collate_fn=collate_fn,
    drop_last=True)

for _ in tqdm(train_data_loader, smoothing=0.1):
    pass
