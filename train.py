import torch.utils.data
# import augmentation
import sklearn.metrics
# import losses
import torch.nn as nn
import torchvision
from ticpfptp.torch import fix_seed, load_weights, save_model
from ticpfptp.metrics import Mean
from ticpfptp.format import args_to_string, args_to_path
# import utils
from tensorboardX import SummaryWriter
import os
import logging
import numpy as np
import argparse
from tqdm import tqdm
from dataset import TrainEvalDataset
from model import Model
import torch.nn.functional as F


# TODO: pack padded seq for targets

def compute_score(labels, logits):
    true = labels.data.cpu().numpy()
    pred = (logits > 0.).data.cpu().numpy()
    score = sklearn.metrics.f1_score(y_true=true, y_pred=pred, average='macro')

    return score


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
    # TODO:
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
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--sched', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    logging.info(args_to_string(args))
    experiment_path = os.path.join(args.experiment_path, args_to_path(args))
    fix_seed(args.seed)

    train_dataset = TrainEvalDataset(args.dataset_path, subset='train-clean-100')
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn)

    eval_dataset = TrainEvalDataset(args.dataset_path, subset='dev-clean')
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
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
    metrics = {'loss': Mean()}

    for epoch in range(args.epochs):
        if epoch % 10 == 0:
            logging.info(experiment_path)

        model.train()
        for (spectras, spectras_mask), (labels, labels_mask) in tqdm(
                train_data_loader, desc='epoch {} training'.format(epoch)):
            spectras, spectras_mask = spectras.to(device), spectras_mask.to(device)
            # labels, labels_mask = labels.to(device), labels_mask.to(device)
            # logits = model(spectras, labels[:, :-1])
            #
            # loss = compute_loss(labels=labels[:, 1:], logits=logits, mask=labels_mask[:, 1:])
            # metrics['loss'].update(loss.data.cpu().numpy())
            #
            # optimizer.zero_grad()
            # loss.mean().backward()
            # optimizer.step()

        train_writer.add_scalar('loss', metrics['loss'].compute_and_reset(), global_step=epoch)
        train_writer.add_scalar(
            'learning_rate',
            np.squeeze([param_group['lr'] for param_group in optimizer.param_groups]),
            global_step=epoch)
        # train_writer.add_image(
        #     'images', torchvision.utils.make_grid(images.sigmoid().cpu()), global_step=epoch)

        model.eval()
        with torch.no_grad():
            all_labels = []
            all_logits = []

            for (spectras, spectras_mask), (labels, labels_mask) in tqdm(
                    eval_data_loader, desc='epoch {} evaluating'.format(epoch)):
                spectras, spectras_mask = spectras.to(device), spectras_mask.to(device)
                labels, labels_mask = labels.to(device), labels_mask.to(device)
                logits = model(spectras, labels[:, :-1])

                loss = compute_loss(labels=labels[:, 1:], logits=logits, mask=labels_mask[:, 1:])
                metrics['loss'].update(loss.data.cpu().numpy())

                all_labels.append(labels)
                all_logits.append(logits)

            all_labels = torch.cat(all_labels, 0)
            all_logits = torch.cat(all_logits, 0)

        eval_loss = metrics['loss'].compute_and_reset()
        eval_score = compute_score(labels=all_labels, logits=all_logits)
        eval_writer.add_scalar('loss', eval_loss, global_step=epoch)
        eval_writer.add_scalar('score', eval_score, global_step=epoch)

        save_model(model_to_save, experiment_path)
        # utils.save_model(model_to_save, utils.mkdir(os.path.join(experiment_path, 'epoch_{}'.format(epoch))))
        # if eval_score > best_score:
        #     best_score = eval_score
        #     save_model(model_to_save, mkdir(os.path.join(experiment_path, 'best')))

        scheduler.step(eval_loss)


if __name__ == '__main__':
    main()
