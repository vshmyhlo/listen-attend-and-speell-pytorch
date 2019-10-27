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
from tensorboardX import SummaryWriter
from termcolor import colored
from ticpfptp.format import args_to_string
from ticpfptp.metrics import Mean
from ticpfptp.os import mkdir
from ticpfptp.torch import fix_seed, load_weights, save_model
from tqdm import tqdm

from dataset import TrainEvalDataset, SAMPLE_RATE, load_data
from metrics import word_error_rate
from model import Model
from sampler import BatchSampler
from transforms import LoadSignal, ApplyTo, Extract, VocabEncode, ToTensor
from vocab import CHAR_VOCAB, CharVocab


# TODO: preemphasis?
# TODO: configure attention type
# TODO: use import scipy.io.wavfile as wav
# TODO: bucketing
# TODO: transformer loss sum
# TODO: normalization, spectra computing, number of features (freq)
# TODO: warmup
# TODO: dropout
# TODO: check targets are correct
# TODO: pack sequence
# TODO: per freq norm
# TODO: mask attention
# TODO: log ignore keys
# TODO: pack padded seq for targets
# TODO: min or max score scheduling
# TODO: mask attention
# TODO: loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')


def take_until_token(seq, token):
    if token in seq:
        return seq[:seq.index(token)]
    else:
        return seq


# TODO: check correct truncation
def compute_wer(input, target, vocab, pool):
    pred = np.argmax(input.data.cpu().numpy(), -1)
    true = target.data.cpu().numpy()

    hyps = [take_until_token(pred.tolist(), vocab.eos_id) for pred in pred]
    refs = [take_until_token(true.tolist(), vocab.eos_id) for true in true]

    hyps = map(lambda hyp: vocab.decode(hyp).split(), hyps)
    refs = map(lambda ref: vocab.decode(ref).split(), refs)
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
    loss = softmax_cross_entropy(input=input, target=target, axis=2)
    loss = (loss * mask.float()).sum(1)

    return loss


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-path', type=str, default='./tf_log')
    parser.add_argument('--restore-path', type=str)
    parser.add_argument('--dataset-path', type=str, default='./data')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--opt', type=str, choices=['adam', 'momentum'], default='adam')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lab-smooth', type=float, default=0.1)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--sched', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    logging.info(args_to_string(args))
    fix_seed(args.seed)

    train_data = pd.concat([
        load_data(os.path.join(args.dataset_path, 'train-clean-100'), workers=args.workers),
        load_data(os.path.join(args.dataset_path, 'train-clean-360'), workers=args.workers),
    ])
    eval_data = pd.concat([
        load_data(os.path.join(args.dataset_path, 'dev-clean'), workers=args.workers),
    ])

    vocab = CharVocab(CHAR_VOCAB)
    # vocab = WordVocab(train_data['syms'])
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
        batch_sampler=BatchSampler(train_data, batch_size=args.bs, shuffle=True, drop_last=True),
        num_workers=args.workers,
        collate_fn=collate_fn)

    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_sampler=BatchSampler(eval_data, batch_size=args.bs),
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

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    elif args.opt == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise AssertionError('invalid optimizer {}'.format(args.opt))

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=args.sched)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500 * args.epochs)

    # main loop
    train_writer = SummaryWriter(os.path.join(args.experiment_path, 'train'))
    eval_writer = SummaryWriter(os.path.join(args.experiment_path, 'eval'))
    best_wer = float('inf')

    for epoch in range(args.epochs):
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
                itertools.islice(train_data_loader, 500), total=500, desc='epoch {} training'.format(epoch),
                smoothing=0.01):
            sigs, labels = sigs.to(device), labels.to(device)
            sigs_mask, labels_mask = sigs_mask.to(device), labels_mask.to(device)

            logits, etc = model(sigs, sigs_mask, labels[:, :-1])

            loss = compute_loss(
                input=logits, target=labels[:, 1:], mask=labels_mask[:, 1:], smoothing=args.lab_smooth)
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
            train_writer.add_image(
                'weights',
                torchvision.utils.make_grid(etc['weights'], nrow=compute_nrow(etc['weights']), normalize=True),
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
                    input=logits, target=labels[:, 1:], mask=labels_mask[:, 1:], smoothing=args.lab_smooth)
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
            eval_writer.add_image(
                'weights',
                torchvision.utils.make_grid(etc['weights'], nrow=compute_nrow(etc['weights']), normalize=True),
                global_step=epoch)

        save_model(model_to_save, args.experiment_path)
        # utils.save_model(model_to_save, utils.mkdir(os.path.join(experiment_path, 'epoch_{}'.format(epoch))))
        if metrics['wer'] < best_wer:
            best_wer = metrics['wer']
            save_model(model_to_save, mkdir(os.path.join(args.experiment_path, 'best')))

        # scheduler.step(eval_loss)


if __name__ == '__main__':
    main()
