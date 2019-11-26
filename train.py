import logging
import math
import os
import time
from multiprocessing import Pool

import click
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
from utils import take_until_token, label_smoothing, one_hot
from vocab import SubWordVocab, CHAR_VOCAB, CharVocab, WordVocab


# TODO: use decoder infer method for eval
# TODO: relative positional encoding
# TODO: Minimum Word Error Rate (MWER) Training
# TODO: Scheduled Sampling
# TODO: better tokenization for word level model
# TODO: layer norm
# TODO: check targets are correct
# TODO: per freq norm
# TODO: norm spec 1d
# TODO: better loss averaging
# TODO: beam search


def draw_attention(weights):
    color = np.random.RandomState(42).uniform(0.5, 1, size=(1, weights.size(1), 3, 1, 1))
    color = torch.tensor(color, dtype=weights.dtype, device=weights.device)

    weights = weights.unsqueeze(2)
    images = (weights * color).mean(1)

    return images


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


def softmax_cross_entropy(input, target, axis=1, keepdim=False):
    log_prob = input.log_softmax(axis)
    loss = -(target * log_prob).sum(axis, keepdim=keepdim)

    return loss


def compute_loss(input, target, mask, smoothing):
    target = one_hot(target, input.size(2))
    target = label_smoothing(target, smoothing)

    loss = softmax_cross_entropy(input=input, target=target, axis=2)
    loss = (loss * mask.float()).sum(1)

    return loss


@click.command()
@click.option('--experiment-path', type=click.Path(), default='./tf_log')
@click.option('--dataset-path', type=click.Path(), default='./data')
@click.option('--config-path', type=click.Path(), required=True)
@click.option('--restore-path', type=click.Path())
@click.option('--workers', type=click.INT, default=os.cpu_count())
def main(experiment_path, dataset_path, config_path, restore_path, workers):
    logging.basicConfig(level=logging.INFO)
    config = Config.from_json(config_path)
    fix_seed(config.seed)

    train_data = pd.concat([
        load_data(os.path.join(dataset_path, 'train-clean-100'), workers=workers),
        load_data(os.path.join(dataset_path, 'train-clean-360'), workers=workers),
    ])
    eval_data = pd.concat([
        load_data(os.path.join(dataset_path, 'dev-clean'), workers=workers),
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
        num_workers=workers,
        collate_fn=collate_fn)

    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_sampler=BatchSampler(eval_data, batch_size=config.batch_size),
        num_workers=workers,
        collate_fn=collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(SAMPLE_RATE, len(vocab))
    model_to_save = model
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    if restore_path is not None:
        load_weights(model_to_save, restore_path)

    if config.opt.type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), config.opt.lr, weight_decay=1e-4)
    elif config.opt.type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), config.opt.lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise AssertionError('invalid config.opt.type {}'.format(config.opt.type))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_data_loader) * config.epochs)

    # ==================================================================================================================
    # main loop

    train_writer = SummaryWriter(os.path.join(experiment_path, 'train'))
    eval_writer = SummaryWriter(os.path.join(experiment_path, 'eval'))
    best_wer = float('inf')

    for epoch in range(config.epochs):
        if epoch % 10 == 0:
            logging.info(experiment_path)

        # ==============================================================================================================
        # training

        metrics = {
            'loss': Mean(),
            'fps': Mean(),
        }

        model.train()
        t1 = time.time()
        for (sigs, labels), (sigs_mask, labels_mask) in tqdm(
                train_data_loader,
                desc='epoch {} training'.format(epoch),
                smoothing=0.01):
            sigs, labels = sigs.to(device), labels.to(device)
            sigs_mask, labels_mask = sigs_mask.to(device), labels_mask.to(device)

            logits, etc = model(sigs, labels[:, :-1], sigs_mask, labels_mask[:, :-1])

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
            for k in etc['weights']:
                w = etc['weights'][k]
                train_writer.add_image(
                    'weights/{}'.format(k),
                    torchvision.utils.make_grid(w, nrow=compute_nrow(w), normalize=True),
                    global_step=epoch)

            for i, (true, pred) in enumerate(zip(
                    labels[:, 1:][:4].detach().data.cpu().numpy(),
                    np.argmax(logits[:4].detach().data.cpu().numpy(), -1))):
                print('{}:'.format(i))
                text = vocab.decode(take_until_token(true.tolist(), vocab.eos_id))
                print(colored(text, 'green'))
                text = vocab.decode(take_until_token(pred.tolist(), vocab.eos_id))
                print(colored(text, 'yellow'))

        # ==============================================================================================================
        # evaluation

        metrics = {
            # 'loss': Mean(),
            'wer': Mean(),
        }

        model.eval()
        with torch.no_grad(), Pool(workers) as pool:
            for (sigs, labels), (sigs_mask, labels_mask) in tqdm(
                    eval_data_loader, desc='epoch {} evaluating'.format(epoch), smoothing=0.1):
                sigs, labels = sigs.to(device), labels.to(device)
                sigs_mask, labels_mask = sigs_mask.to(device), labels_mask.to(device)

                logits, etc = model(sigs, labels[:, :-1], sigs_mask, labels_mask[:, :-1])
                # logits, etc = model.infer(
                #     sigs, sigs_mask, sos_id=vocab.sos_id, eos_id=vocab.eos_id, max_steps=labels.size(1) + 10)

                # loss = compute_loss(
                #     input=logits, target=labels[:, 1:], mask=labels_mask[:, 1:], smoothing=config.label_smoothing)
                # metrics['loss'].update(loss.data.cpu().numpy())

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
            for k in etc['weights']:
                w = etc['weights'][k]
                eval_writer.add_image(
                    'weights/{}'.format(k),
                    torchvision.utils.make_grid(w, nrow=compute_nrow(w), normalize=True),
                    global_step=epoch)

        save_model(model_to_save, experiment_path)
        if metrics['wer'] < best_wer:
            best_wer = metrics['wer']
            save_model(model_to_save, mkdir(os.path.join(experiment_path, 'best')))


if __name__ == '__main__':
    main()
