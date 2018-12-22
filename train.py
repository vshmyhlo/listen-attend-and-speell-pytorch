import torch.utils.data
from multiprocessing import Pool
import torch.nn as nn
from ticpfptp.torch import fix_seed, load_weights, save_model
from ticpfptp.metrics import Mean
from ticpfptp.format import args_to_string, args_to_path
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


# TODO: log ignore keys
# TODO: pack padded seq for targets
# TODO: min or max score scheduling
# TODO: CER, WER, paralellize

def take_until_token(seq, token):
    if token in seq:
        return seq[:seq.index(token)]
    else:
        return seq


# def compute_score(labels, logits, vocab):
#     true = labels.data.cpu().numpy()
#     logits = logits.data.cpu().numpy()
#     pred = np.argmax(logits, -1)
#     wers = [
#         word_error_rate(
#             ref=take_until_token(true.tolist(), vocab.eos_id),
#             hyp=take_until_token(pred.tolist(), vocab.eos_id))
#         for true, pred in zip(true, pred)]
#
#     return wers

def chars_to_words(seq):
    return ''.join(seq).split(' ')


# TODO: check correct truncation
def compute_score(labels, logits, vocab, pool):
    true = labels.data.cpu().numpy()
    logits = logits.data.cpu().numpy()
    pred = np.argmax(logits, -1)

    refs = [take_until_token(true.tolist(), vocab.eos_id) for true in true]
    hyps = [take_until_token(pred.tolist(), vocab.eos_id) for pred in pred]
    cers = pool.starmap(word_error_rate, zip(refs, hyps))

    refs = map(lambda ref: chars_to_words(vocab.decode(ref)), refs)
    hyps = map(lambda hyp: chars_to_words(vocab.decode(hyp)), hyps)
    wers = pool.starmap(word_error_rate, zip(refs, hyps))

    return cers, wers


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

    assert np.array_equal(spectras, spectras * np.expand_dims(spectras_mask, -1))
    assert np.array_equal(seqs, seqs * seqs_mask)

    spectras, spectras_mask = torch.from_numpy(spectras).float(), torch.from_numpy(spectras_mask).byte()
    seqs, seqs_mask = torch.from_numpy(seqs).long(), torch.from_numpy(seqs_mask).byte()

    return (spectras, spectras_mask), (seqs, seqs_mask)


def compute_loss(labels, logits, mask):
    # TODO: sum over time or mean over time?
    labels = labels[mask]
    logits = logits[mask]
    loss = F.cross_entropy(input=logits, target=labels, reduction='none')

    return loss


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-path', type=str, default='./tf_log')
    parser.add_argument('--restore-path', type=str)
    parser.add_argument('--dataset-path', type=str, default='./data')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--opt', type=str, choices=['adam', 'momentum'], default='momentum')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
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
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn)

    eval_dataset = TrainEvalDataset(args.dataset_path, subset='dev-clean')
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(len(train_dataset.vocab))
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=args.sched)

    # training
    train_writer = SummaryWriter(experiment_path)
    eval_writer = SummaryWriter(os.path.join(experiment_path, 'eval'))
    best_score = 0

    # metrics
    metrics = {'loss': Mean(), 'cer': Mean(), 'wer': Mean()}

    for epoch in range(args.epochs):
        if epoch % 10 == 0:
            logging.info(experiment_path)

        model.train()
        for (spectras, spectras_mask), (labels, labels_mask) in tqdm(
                train_data_loader, desc='epoch {} training'.format(epoch)):
            spectras, spectras_mask = spectras.to(device), spectras_mask.to(device)
            labels, labels_mask = labels.to(device), labels_mask.to(device)
            logits = model(spectras, labels[:, :-1])

            loss = compute_loss(labels=labels[:, 1:], logits=logits, mask=labels_mask[:, 1:])
            metrics['loss'].update(loss.data.cpu().numpy())

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            break

        train_writer.add_scalar('loss', metrics['loss'].compute_and_reset(), global_step=epoch)
        train_writer.add_scalar(
            'learning_rate',
            np.squeeze([param_group['lr'] for param_group in optimizer.param_groups]),
            global_step=epoch)
        # train_writer.add_image(
        #     'images', torchvision.utils.make_grid(images.sigmoid().cpu()), global_step=epoch)
        # train_writer.add_audio()

        model.eval()
        with torch.no_grad(), Pool(args.workers) as pool:
            for (spectras, spectras_mask), (labels, labels_mask) in tqdm(
                    eval_data_loader, desc='epoch {} evaluating'.format(epoch)):
                spectras, spectras_mask = spectras.to(device), spectras_mask.to(device)
                labels, labels_mask = labels.to(device), labels_mask.to(device)
                logits = model(spectras, labels[:, :-1])

                loss = compute_loss(labels=labels[:, 1:], logits=logits, mask=labels_mask[:, 1:])
                metrics['loss'].update(loss.data.cpu().numpy())

                cer, wer = compute_score(labels=labels[:, 1:], logits=logits, vocab=train_dataset.vocab, pool=pool)
                metrics['cer'].update(cer)
                metrics['wer'].update(wer)

        eval_loss = metrics['loss'].compute_and_reset()
        # eval_score = metrics['score'].compute_and_reset()
        eval_writer.add_scalar('loss', eval_loss, global_step=epoch)
        # eval_writer.add_scalar('score', eval_score, global_step=epoch)

        save_model(model_to_save, experiment_path)
        # utils.save_model(model_to_save, utils.mkdir(os.path.join(experiment_path, 'epoch_{}'.format(epoch))))
        # if eval_score > best_score:
        #     best_score = eval_score
        #     save_model(model_to_save, mkdir(os.path.join(experiment_path, 'best')))

        scheduler.step(eval_loss)


if __name__ == '__main__':
    main()
