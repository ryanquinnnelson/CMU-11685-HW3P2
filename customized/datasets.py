"""
Contains all Dataset objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def collate_fn_trainval(batch):
    """
    Format batch of training records for use in training phase or batch of validation records for use in validation phase.
    :param batch (Tuple(Tensor,Tensor)): records to be formatted. Has shape x=(N_TIMESTEPS,FEATURES),y=(UTTERANCE_LABEL_LENGTH,)
    :return: Tuple(Tensor,Tensor,Tensor,Tensor) representing (batch_x, batch_y, lengths_x, lengths_y)
    """
    # sort batch by decreasing sequence length for efficient packing
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    # split batch into features and target
    batch_x = [x for x, y in batch]  # List of x Tensors, len(batch_x)=BATCHSIZE
    lengths_x = torch.LongTensor([len(x) for x, y in batch])
    batch_y = [y for x, y in batch]  # List of y Tensors, len(batch_y)=BATCHSIZE
    lengths_y = torch.LongTensor([len(y) for x, y in batch])

    # Pad sequences to have the same number of rows per utterance
    # CTCLoss expects batch second, but batch first reduces number of tranpositions in the forward pass
    batch_x = pad_sequence(batch_x, batch_first=True)  # (BATCHSIZE,MAX_N_TIMESTEPS,FEATURES)

    # Pad targets to have the same number of elements per utterance
    # CTCLoss expects batch first for targets
    batch_y = pad_sequence(batch_y, batch_first=True)  # (BATCHSIZE,MAX_UTTERANCE_LABEL_LENGTH)

    return batch_x, batch_y, lengths_x, lengths_y


class TrainValDataset(Dataset):
    """
    Define a Dataset for training and validation data.
    """

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


def collate_fn_test(batch):
    """
    Format batch of records for use in testing phase.

    :param batch (Tensor): records to be formatted. Has shape (N_TIMESTEPS,FEATURES)
    :return: Tuple(Tensor,Tensor) representing (batch_x, lengths_x)
    """
    lengths_x = torch.LongTensor([len(x) for x in batch])

    # Pad sequences to have the same number of rows per utterance
    batch = pad_sequence(batch, batch_first=True)

    return batch, lengths_x


class TestDataset(Dataset):
    """
    Define Dataset for test data.
    """

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
