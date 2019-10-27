import os

import pandas as pd
import torchvision.transforms as T

from dataset import load_data, TrainEvalDataset, SAMPLE_RATE
from transforms import ApplyTo, LoadSignal, Extract


def main(dataset_path):
    transform = T.Compose([
        ApplyTo(['sig'], T.Compose([
            LoadSignal(SAMPLE_RATE),
        ])),
        Extract(['sig', 'syms']),

    ])

    data = pd.concat([
        load_data(os.path.join(dataset_path, 'train-clean-100')),
        load_data(os.path.join(dataset_path, 'train-clean-360')),
        load_data(os.path.join(dataset_path, 'dev-clean')),
    ])

    dataset = TrainEvalDataset(data, transform=transform)


if __name__ == '__main__':
    main()
