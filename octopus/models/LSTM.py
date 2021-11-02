"""
All things related to LSTMs.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, bidirectional, dropout):
        super(BasicLSTM, self).__init__()

        # TODO: cnn could go here (40 -> 128, 256 output channels) feature extractor
        # TODO: kaiming initialization for cnn
        # TODO: batch normalization for cnn

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=False,  # faster if we don't do batch first
                           dropout=dropout,
                           bidirectional=bidirectional)

        self.linear = nn.Linear(hidden_size, output_size)
        # TODO: xavier initialization for linear
        # TODO: could do batch normalization after linear

    def forward(self, x, lengths_x):
        # pack sequence after any cnn layers
        # x: (N_TIMESTEPS x BATCHSIZE x FEATURES)
        packed_x = pack_padded_sequence(x, lengths_x, enforce_sorted=True)

        # TODO: initialize hidden layer and cell state layer -  look into this
        # out: (N_TIMESTEPS x BATCHSIZE x HIDDEN_SIZE * DIRECTIONS)
        # h_t: (DIRECTIONS x BATCHSIZE x HIDDEN_SIZE)  DIRECTIONS=2 for bidirectional, 1 otherwise
        # c_t: (DIRECTIONS x BATCHSIZE x HIDDEN_SIZE)  DIRECTIONS=2 for bidirectional, 1 otherwise
        out, (h_t, c_t) = self.rnn(packed_x)

        # unpack sequence for use in linear layer
        # unpacked_x: (N_TIMESTEPS x BATCHSIZE x HIDDEN_SIZE * DIRECTIONS)
        # lengths_x: (BATCHSIZE,)
        unpacked_out, lengths_out = pad_packed_sequence(out, batch_first=False)

        # out: (N_TIMESTEPS x BATCHSIZE x N_LABELS)
        linear_out = self.linear(unpacked_out)

        out = nn.functional.log_softmax(linear_out, dim=2)  # N_LABELS is the dimension to softmax

        logging.info('--forward--')
        logging.info(f'x_t:{x.shape},lengths_x:{lengths_x}')
        logging.info(f'h_t:{h_t.shape}')
        logging.info(f'c_t:{c_t.shape}')
        logging.info(f'out_t:{unpacked_out.shape}, lengths_out:{lengths_out}')
        logging.info(f'linear:{linear_out.shape}')
        logging.info(f'softmax:{out.shape}')
        logging.info('')
        return out
