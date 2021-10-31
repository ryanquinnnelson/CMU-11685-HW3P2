"""
Contains all Dataset objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset


def trainval_collate_fn(batch):
    # Pad sequences to have the same number of rows per utterance
    # print(type(batch))
    # print(len(batch))
    print('--collate--')
    for i, b in enumerate(batch):
        print(f'x_{i}', b[0].shape, f'y_{i}', b[1].shape)

    # sort batch by decreasing sequence length
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    # split batch into features and target
    ### Select all data from batch (1 line)
    batch_x = [x for x, y in batch]
    # for b in batch_x:
    #   print('xb',b.shape)
    # print()
    lengths_x = torch.LongTensor([len(x) for x, y in batch])

    ### Select all labels from batch (1 line)
    batch_y = [y for x, y in batch]
    # for b in batch_y:
    #   print('yb', b.shape)
    lengths_y = torch.LongTensor([len(y) for x, y in batch])
    print('lengths_x', lengths_x, 'lengths_y', lengths_y)

    # pad sequence and convert to tensor
    pad_batch_x = pad_sequence(batch_x)  # CTCLoss expects batch second for input
    print('pad_batch_x', pad_batch_x.shape)
    # print('paddedx',pad_batch_x.size())

    # ?? do we pad and pack targets
    # pad targets and convert to tensor
    pad_batch_y = pad_sequence(batch_y, batch_first=True)  # CTCLoss expects batch first for targets
    # print('paddedy',pad_batch_y.size())
    print('pad_batch_y', pad_batch_y.shape)

    # pack sequence
    print('packed_batch_x')
    print()
    packed_batch_x = pack_padded_sequence(pad_batch_x, lengths_x, enforce_sorted=True)
    # packed_batch_y = pack_padded_sequence(pad_batch_y, lengths_y, batch_first=True,enforce_sorted=False)

    # # unpack debug
    # seq_unpackedx, lens_unpackedx = pad_packed_sequence(packed_batch_x, batch_first=True)
    # print('unpackedx',seq_unpackedx.shape)
    # print('unpacked_lengthsx',lens_unpackedx)
    # print('returned\n\n')
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


def test_collate_fn(batch):
    # TODO: Pad X
    # print(type(batch))
    # print(len(batch))
    # # print(batch[:1])
    # print(type(batch[0]))
    # print(len(batch[0]))
    # print()

    # sort batch by decreasing sequence length
    batch = sorted(batch, key=lambda x: len(x), reverse=True)

    lengths_x = torch.LongTensor([len(x) for x in batch])
    # print(lengths_x)

    # pad sequence and convert to tensor
    pad_batch_x = pad_sequence(batch)
    # print('paddedx',pad_batch_x.size())

    # pack sequence
    packed_batch_x = pack_padded_sequence(pad_batch_x, lengths_x, enforce_sorted=True)

    # # unpack debug
    # seq_unpackedx, lens_unpackedx = pad_packed_sequence(packed_batch_x, batch_first=True)
    # print('unpackedx',seq_unpackedx.shape)
    # print('unpacked_lengthsx',lens_unpackedx)
    # print('returned\n\n')
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
