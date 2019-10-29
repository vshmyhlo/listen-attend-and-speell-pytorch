import argparse
import itertools
import logging
import math
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.config import Config
from all_the_tools.metrics import Mean
from tensorboardX import SummaryWriter
from termcolor import colored
from ticpfptp.os import mkdir
from ticpfptp.torch import fix_seed, load_weights, save_model
from tqdm import tqdm

from dataset import TrainEvalDataset, SAMPLE_RATE, load_data
from metrics import word_error_rate
from model import Model
from sampler import BatchSampler
from transforms import LoadSignal, ApplyTo, Extract, VocabEncode, ToTensor
from utils import take_until_token
from vocab import SubWordVocab, CHAR_VOCAB, CharVocab, WordVocab

# TODO: word-piece model / subword model
# TODO: multi-head attention
# TODO: Minimum Word Error Rate (MWER) Training
# TODO: Scheduled Sampling
# TODO: sgd
# TODO: better tokenization for word level model
# TODO: layer norm
# TODO: use import scipy.io.wavfile as wav
# TODO: transformer loss sum
# TODO: encoder/decoder self-attention
# TODO: positional encoding
# TODO: warmup
# TODO: dropout
# TODO: residual
# TODO: check targets are correct
# TODO: pack sequence
# TODO: per freq norm
# TODO: pack padded seq for targets
# TODO: better loss averaging
# TODO: loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')


N = 1000


# TODO: check correct truncation
def compute_wer(input, target, vocab, pool):
    pred = np.argmax(input.data.cpu().numpy(), -1)
    true = target.data.cpu().numpy()

    hyps = [take_until_token(pred.tolist(), vocab.eos_id) for pred in pred]
    refs = [take_until_token(true.tolist(), vocab.eos_id) for true in true]

    hyps = [vocab.decode(hyp).split() for hyp in hyps]
    refs = [vocab.decode(ref).split() for ref in refs]

    wers = pool.starmap(word_error_rate, zip(refs, hyps))

    return wers


def compute_nrow(images):
    b, _, h, w = images.size()
    nrow = math.ceil(math.sqrt(h * b / w))

    return nrow


def pad_and_pack(tensors):
    sizes = [t.shape[0] for t in tensors]

    tensor = torch.zeros(
        len(sizes), max(sizes), dtype=tensors[0].dtype, layout=tensors[0].layout, device=tensors[0].device)
    mask = torch.zeros(
        len(sizes), max(sizes), dtype=torch.bool, layout=tensors[0].layout, device=tensors[0].device)

    for i, t in enumerate(tensors):
        tensor[i, :t.size(0)] = t
        mask[i, :t.size(0)] = True

    return tensor, mask


def collate_fn(batch):
    sigs, syms = list(zip(*batch))

    sigs, sigs_mask = pad_and_pack(sigs)
    syms, syms_mask = pad_and_pack(syms)

    return (sigs, syms), (sigs_mask, syms_mask)


def one_hot(input, num_classes):
    input = torch.eye(num_classes, dtype=torch.float, device=input.device)[input]

    return input


def softmax_cross_entropy(input, target, axis=1, keepdim=False):
    log_prob = input.log_softmax(axis)
    loss = -(target * log_prob).sum(axis, keepdim=keepdim)

    return loss


def compute_loss(input, target, mask, smoothing):
    target = one_hot(target, input.size(2))
    target = target * (1 - smoothing) + smoothing / input.size(2)

    loss = softmax_cross_entropy(input=input, target=target, axis=2)
    loss = (loss * mask.float()).sum(1)

    return loss


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-path', type=str, default='./tf_log')
    parser.add_argument('--dataset-path', type=str, default='./data')
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--restore-path', type=str)
    parser.add_argument('--workers', type=int, default=os.cpu_count())

    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    config = Config.from_json(args.config_path)
    fix_seed(config.seed)

    train_data = pd.concat([
        load_data(os.path.join(args.dataset_path, 'train-clean-100'), workers=args.workers),
        load_data(os.path.join(args.dataset_path, 'train-clean-360'), workers=args.workers),
    ])
    eval_data = pd.concat([
        load_data(os.path.join(args.dataset_path, 'dev-clean'), workers=args.workers),
    ])

    if config.vocab == 'char':
        vocab = CharVocab(CHAR_VOCAB)
    elif config.vocab == 'word':
        vocab = WordVocab(train_data['syms'], 30000)
    elif config.vocab == 'subword':
        vocab = SubWordVocab(10000)
    else:
        raise AssertionError('invalid config.vocab: {}'.format(config.vocab))

    train_transform = T.Compose([
        ApplyTo(['sig'], T.Compose([
            LoadSignal(SAMPLE_RATE),
            ToTensor(),
        ])),
        ApplyTo(['syms'], T.Compose([
            VocabEncode(vocab),
            ToTensor(),
        ])),
        Extract(['sig', 'syms']),

    ])
    eval_transform = train_transform

    train_dataset = TrainEvalDataset(train_data, transform=train_transform)
    eval_dataset = TrainEvalDataset(eval_data, transform=eval_transform)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=BatchSampler(train_data, batch_size=config.batch_size, shuffle=True, drop_last=True),
        num_workers=args.workers,
        collate_fn=collate_fn)

    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_sampler=BatchSampler(eval_data, batch_size=config.batch_size),
        num_workers=args.workers,
        collate_fn=collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(SAMPLE_RATE, len(vocab))
    model_to_save = model
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    if args.restore_path is not None:
        load_weights(model_to_save, args.restore_path)

    if config.opt.type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), config.opt.lr, weight_decay=1e-4)
    elif config.opt.type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), config.opt.lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise AssertionError('invalid config.opt.type {}'.format(config.opt.type))
    # optimizer = LA(optimizer, lr=0.5, num_steps=5)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=args.sched)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N * config.epochs)

    # main loop
    train_writer = SummaryWriter(os.path.join(args.experiment_path, 'train'))
    eval_writer = SummaryWriter(os.path.join(args.experiment_path, 'eval'))
    best_wer = float('inf')

    for epoch in range(config.epochs):
        if epoch % 10 == 0:
            logging.info(args.experiment_path)

        # training
        metrics = {
            'loss': Mean(),
            'fps': Mean(),
        }

        model.train()
        t1 = time.time()
        for (sigs, labels), (sigs_mask, labels_mask) in tqdm(
                itertools.islice(train_data_loader, N), total=N, desc='epoch {} training'.format(epoch),
                smoothing=0.01):
            sigs, labels = sigs.to(device), labels.to(device)
            sigs_mask, labels_mask = sigs_mask.to(device), labels_mask.to(device)

            logits, etc = model(sigs, sigs_mask, labels[:, :-1])

            loss = compute_loss(
                input=logits, target=labels[:, 1:], mask=labels_mask[:, 1:], smoothing=config.label_smoothing)
            metrics['loss'].update(loss.data.cpu().numpy())

            lr = np.squeeze(scheduler.get_lr())

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            scheduler.step()

            t2 = time.time()
            metrics['fps'].update(1 / ((t2 - t1) / sigs.size(0)))
            t1 = t2

        with torch.no_grad():
            metrics = {k: metrics[k].compute_and_reset() for k in metrics}
            print('[EPOCH {}][TRAIN] {}'.format(
                epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
            for k in metrics:
                train_writer.add_scalar(k, metrics[k], global_step=epoch)
            train_writer.add_scalar('learning_rate', lr, global_step=epoch)

            train_writer.add_image(
                'spectras',
                torchvision.utils.make_grid(etc['spectras'], nrow=compute_nrow(etc['spectras']), normalize=True),
                global_step=epoch)
            for i in range(etc['weights'].size(1)):
                train_writer.add_image(
                    'weights/{}'.format(i),
                    torchvision.utils.make_grid(
                        etc['weights'][:, i:i + 1], nrow=compute_nrow(etc['weights']), normalize=True),
                    global_step=epoch)

            for i, (true, pred) in enumerate(zip(
                    labels[:, 1:][:4].detach().data.cpu().numpy(),
                    np.argmax(logits[:4].detach().data.cpu().numpy(), -1))):
                print('{}:'.format(i))
                text = vocab.decode(take_until_token(true.tolist(), vocab.eos_id))
                print(colored(text, 'green'))
                text = vocab.decode(take_until_token(pred.tolist(), vocab.eos_id))
                print(colored(text, 'yellow'))

        # evaluation
        metrics = {
            'loss': Mean(),
            'wer': Mean(),
        }

        model.eval()
        with torch.no_grad(), Pool(args.workers) as pool:
            for (sigs, labels), (sigs_mask, labels_mask) in tqdm(
                    eval_data_loader, desc='epoch {} evaluating'.format(epoch), smoothing=0.1):
                sigs, labels = sigs.to(device), labels.to(device)
                sigs_mask, labels_mask = sigs_mask.to(device), labels_mask.to(device)

                logits, etc = model(sigs, sigs_mask, labels[:, :-1])

                loss = compute_loss(
                    input=logits, target=labels[:, 1:], mask=labels_mask[:, 1:], smoothing=config.label_smoothing)
                metrics['loss'].update(loss.data.cpu().numpy())

                wer = compute_wer(input=logits, target=labels[:, 1:], vocab=vocab, pool=pool)
                metrics['wer'].update(wer)

        with torch.no_grad():
            metrics = {k: metrics[k].compute_and_reset() for k in metrics}
            print('[EPOCH {}][EVAL] {}'.format(
                epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
            for k in metrics:
                eval_writer.add_scalar(k, metrics[k], global_step=epoch)

            eval_writer.add_image(
                'spectras',
                torchvision.utils.make_grid(etc['spectras'], nrow=compute_nrow(etc['spectras']), normalize=True),
                global_step=epoch)
            for i in range(etc['weights'].size(1)):
                eval_writer.add_image(
                    'weights/{}'.format(i),
                    torchvision.utils.make_grid(
                        etc['weights'][:, i:i + 1], nrow=compute_nrow(etc['weights']), normalize=True),
                    global_step=epoch)

        save_model(model_to_save, args.experiment_path)
        # utils.save_model(model_to_save, utils.mkdir(os.path.join(experiment_path, 'epoch_{}'.format(epoch))))
        if metrics['wer'] < best_wer:
            best_wer = metrics['wer']
            save_model(model_to_save, mkdir(os.path.join(args.experiment_path, 'best')))

        # scheduler.step(eval_loss)


if __name__ == '__main__':
    main()
