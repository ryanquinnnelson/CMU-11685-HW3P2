"""
Contains all Dataset objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset


def collate_fn_trainval(batch):
    # sort batch by decreasing sequence length
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    # split batch into features and target
    batch_x = [x for x, y in batch]
    lengths_x = torch.LongTensor([len(x) for x, y in batch])
    batch_y = [y for x, y in batch]
    lengths_y = torch.LongTensor([len(y) for x, y in batch])

    # Pad sequences to have the same number of rows per utterance
    pad_batch_x = pad_sequence(batch_x, batch_first=False)  # CTCLoss expects batch second for input

    # ?? do we pad and pack targets
    # Pad targets to have the same number of elements per utterance
    pad_batch_y = pad_sequence(batch_y, batch_first=True)  # CTCLoss expects batch first for targets

    # pack sequence
    packed_batch_x = pack_padded_sequence(pad_batch_x, lengths_x, enforce_sorted=True)

    logging.info('--collate--')
    for i, b in enumerate(batch):
        logging.info(f'x_{i}:{b[0].shape}, y_{i}:{b[1].shape}')
    logging.info(f'lengths_x:{lengths_x},lengths_y:{lengths_y}')
    logging.info(f'pad_batch_x:{pad_batch_x.shape}')
    logging.info(f'pad_batch_y:{pad_batch_y.shape}')
    logging.info('packed_batch_x')
    logging.info('')

    return packed_batch_x, pad_batch_y, lengths_x, lengths_y


class TrainValDataset(Dataset):

    # load the dataset
    def __init__(self, x, y):
        # TODO: replace x and y with dataset path and load data from here -> more efficient
        self.X = x
        self.Y = y

    # get number of items/rows in dataset
    def __len__(self):
        return len(self.Y)

    # get row item at some index
    def __getitem__(self, index):
        x = torch.FloatTensor(self.X[index])
        y = torch.LongTensor(self.Y[index])

        return x, y

    # ?? why sort batch


def collate_fn_test(batch):
    # sort batch by decreasing sequence length
    batch = sorted(batch, key=lambda x: len(x), reverse=True)

    lengths_x = torch.LongTensor([len(x) for x in batch])

    # Pad sequences to have the same number of rows per utterance
    pad_batch_x = pad_sequence(batch, batch_first=False)

    # pack sequence
    packed_batch_x = pack_padded_sequence(pad_batch_x, lengths_x, enforce_sorted=True)

    return packed_batch_x, lengths_x


class TestDataset(Dataset):

    # load the dataset
    def __init__(self, x):
        # TODO: replace x dataset path and load data from here -> more efficient
        self.X = x

    # get number of items/rows in dataset
    def __len__(self):
        return len(self.X)

        # get row item at some index

    def __getitem__(self, index):
        x = torch.FloatTensor(self.X[index])
        return x
