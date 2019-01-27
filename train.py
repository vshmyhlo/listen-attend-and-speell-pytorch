import torch.utils.data
import torchvision
from termcolor import colored
from multiprocessing import Pool
import torch.nn as nn
from ticpfptp.torch import fix_seed, load_weights, save_model, one_hot
from ticpfptp.metrics import Mean
from ticpfptp.format import args_to_string, args_to_path
from ticpfptp.os import mkdir
from tensorboardX import SummaryWriter
import os
import logging
import numpy as np
import argparse
from tqdm import tqdm
from dataset import TrainEvalDataset
from model import Model
import torch.nn.functional as F
from metrics import word_error_rate


# TODO: configure attention type
# TODO: use import scipy.io.wavfile as wav
# TODO: transformer loss sum
# TODO: normalization, spectra computing, number of features (freq)
# TODO: warmup
# TODO: word-level
# TODO: dropout
# TODO: check targets are correct
# TODO: pack sequence
# TODO: per freq norm
# TODO: mask attention
# TODO: rescale loss on batch size
# TODO: log ignore keys
# TODO: pack padded seq for targets
# TODO: min or max score scheduling
# TODO: normalize spectras
# TODO: mask attention
# TODO: loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')
# TODO: compute norm mean and std


def take_until_token(seq, token):
    if token in seq:
        return seq[:seq.index(token)]
    else:
        return seq


def chars_to_words(seq):
    return ''.join(seq).split(' ')


# TODO: check correct truncation
def compute_wer(input, target, vocab, pool):
    pred = np.argmax(input.data.cpu().numpy(), -1)
    true = target.data.cpu().numpy()

    hyps = [take_until_token(pred.tolist(), vocab.eos_id) for pred in pred]
    refs = [take_until_token(true.tolist(), vocab.eos_id) for true in true]

    hyps = map(lambda hyp: chars_to_words(vocab.decode(hyp)), hyps)
    refs = map(lambda ref: chars_to_words(vocab.decode(ref)), refs)
    wers = pool.starmap(word_error_rate, zip(refs, hyps))

    return wers


def pad_and_pack(arrays):
    sizes = [array.shape[0] for array in arrays]
    array_masks = np.zeros((len(sizes), max(sizes)))
    for i, size in enumerate(sizes):
        array_masks[i, :size] = 1

    arrays = np.array(
        [np.concatenate([array, np.zeros((max(sizes) - array.shape[0], *array.shape[1:]), dtype=array.dtype)], 0)
         for array in arrays])

    return arrays, array_masks


def collate_fn(samples):
    spectras, seqs = zip(*samples)

    spectras, spectras_mask = pad_and_pack([spectra.T for spectra in spectras])
    seqs, seqs_mask = pad_and_pack(np.array(seqs))

    spectras, spectras_mask = torch.from_numpy(spectras).float(), torch.from_numpy(spectras_mask).byte()
    seqs, seqs_mask = torch.from_numpy(seqs).long(), torch.from_numpy(seqs_mask).byte()

    return (spectras, spectras_mask), (seqs, seqs_mask)


# def compute_loss(input, target, mask):
#     input = input[mask]
#     target = target[mask]
#     loss = F.cross_entropy(input=input, target=target, reduction='none')
#     weight = mask.float()[mask] / mask.size(0)
#     loss = (loss * weight).sum()
#
#     return loss

# def compute_loss(input, target, mask, smoothing=0.1):
#     input = input[mask]
#     target = target[mask]
#     # target = (1 - smoothing) * target + smoothing * (1 / input.size(-1))
#     loss = F.cross_entropy(input=input, target=target, reduction='sum')
#
#     return loss

def compute_loss(input, target, mask, smoothing):
    input = input[mask]
    target = target[mask]
    num_classes = input.size(-1)
    target = one_hot(target, num_classes)
    target = (1 - smoothing) * target + smoothing * (1 / num_classes)
    loss = -(F.log_softmax(input, -1) * target).sum(-1).sum()

    return loss


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-path', type=str, default='./tf_log')
    parser.add_argument('--restore-path', type=str)
    parser.add_argument('--dataset-path', type=str, default='./data')
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--opt', type=str, choices=['adam', 'momentum'], default='momentum')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--clip-norm', type=float)
    parser.add_argument('--lab-smooth', type=float, default=0.1)
    parser.add_argument('--attn-type', type=str, default='luong', choices=['luong', 'bahdanau'])
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--sched', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    logging.info(args_to_string(args))
    experiment_path = os.path.join(args.experiment_path, args_to_path(
        args, ignore=['experiment_path', 'restore_path', 'dataset_path', 'epochs', 'workers']))
    fix_seed(args.seed)

    train_dataset = TrainEvalDataset(args.dataset_path, subset='train-clean-100')
    # train_dataset = TrainEvalDataset(args.dataset_path, subset='dev-clean')
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        drop_last=True)

    eval_dataset = TrainEvalDataset(args.dataset_path, subset='dev-clean')
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(80, args.size, len(train_dataset.vocab))
    model_to_save = model
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    if args.restore_path is not None:
        load_weights(model_to_save, args.restore_path)

    lr = args.lr / 32 / 32 * args.bs
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    elif args.opt == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise AssertionError('invalid optimizer {}'.format(args.opt))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=args.sched)

    # training
    train_writer = SummaryWriter(experiment_path)
    eval_writer = SummaryWriter(os.path.join(experiment_path, 'eval'))
    best_wer = float('inf')

    # metrics
    metrics = {'loss': Mean(), 'wer': Mean()}

    for epoch in range(args.epochs):
        if epoch % 10 == 0:
            logging.info(experiment_path)

        model.train()
        for (spectras, spectras_mask), (labels, labels_mask) in tqdm(
                train_data_loader, desc='epoch {} training'.format(epoch), smoothing=0.1):
            spectras, spectras_mask = spectras.to(device), spectras_mask.to(device)

            labels, labels_mask = labels.to(device), labels_mask.to(device)
            logits, weights = model(spectras, spectras_mask, labels[:, :-1])

            loss = compute_loss(
                input=logits, target=labels[:, 1:], mask=labels_mask[:, 1:], smoothing=args.lab_smooth)
            metrics['loss'].update(loss.data.cpu().numpy())

            optimizer.zero_grad()
            loss.mean().backward()
            if args.clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

        train_writer.add_scalar('loss', metrics['loss'].compute_and_reset(), global_step=epoch)
        train_writer.add_scalar(
            'learning_rate',
            np.squeeze([param_group['lr'] for param_group in optimizer.param_groups]),
            global_step=epoch)
        spectras_norm = spectras.permute(0, 2, 1).unsqueeze(1).cpu()
        train_writer.add_image(
            'spectras', torchvision.utils.make_grid(spectras_norm, nrow=1, normalize=True), global_step=epoch)
        train_writer.add_image(
            'weights', torchvision.utils.make_grid(weights.unsqueeze(1).cpu(), nrow=1), global_step=epoch)

        for i, (true, pred) in enumerate(zip(
                labels[:, 1:][:4].detach().data.cpu().numpy(),
                np.argmax(logits[:4].detach().data.cpu().numpy(), -1))):
            print('{}:'.format(i))
            text = ''.join(train_dataset.vocab.decode(take_until_token(true.tolist(), train_dataset.vocab.eos_id)))
            print(colored(text, 'green'))
            text = ''.join(train_dataset.vocab.decode(take_until_token(pred.tolist(), train_dataset.vocab.eos_id)))
            print(colored(text, 'yellow'))

        model.eval()
        with torch.no_grad(), Pool(args.workers) as pool:
            for (spectras, spectras_mask), (labels, labels_mask) in tqdm(
                    eval_data_loader, desc='epoch {} evaluating'.format(epoch), smoothing=0.1):
                spectras, spectras_mask = spectras.to(device), spectras_mask.to(device)
                labels, labels_mask = labels.to(device), labels_mask.to(device)
                logits, _ = model(spectras, spectras_mask, labels[:, :-1])

                loss = compute_loss(
                    input=logits, target=labels[:, 1:], mask=labels_mask[:, 1:], smoothing=args.lab_smooth)
                metrics['loss'].update(loss.data.cpu().numpy())

                wer = compute_wer(input=logits, target=labels[:, 1:], vocab=train_dataset.vocab, pool=pool)
                metrics['wer'].update(wer)

        eval_loss = metrics['loss'].compute_and_reset()
        eval_wer = metrics['wer'].compute_and_reset()
        eval_writer.add_scalar('loss', eval_loss, global_step=epoch)
        eval_writer.add_scalar('wer', eval_wer, global_step=epoch)

        save_model(model_to_save, experiment_path)
        # utils.save_model(model_to_save, utils.mkdir(os.path.join(experiment_path, 'epoch_{}'.format(epoch))))
        if eval_wer < best_wer:
            best_wer = eval_wer
            save_model(model_to_save, mkdir(os.path.join(experiment_path, 'best')))

        scheduler.step(eval_loss)


if __name__ == '__main__':
    main()
