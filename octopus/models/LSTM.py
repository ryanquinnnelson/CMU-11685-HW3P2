"""
All things related to LSTMs.
"""
__author__ = 'ryanquinnnelson'

import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
import logging


# TODO Remove print statements
class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, bidirectional, dropout):
        super(BasicLSTM, self).__init__()

        # embedding layer?

        # batch normalization?

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=False,  # faster if we don't do batch first
                           dropout=dropout,
                           bidirectional=bidirectional)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # ?? what was lengths used for here? was it returned in the dataloader too as a third arg?

        # unpacked_x: (N_TIMESTEPS X BATCHSIZE X FEATURES)
        # lengths_x: (BATCHSIZE,)
        unpacked_x, lengths_x = pad_packed_sequence(x, batch_first=False)

        # ?? initialize hidden layer and cell state layer??
        # out: (N_TIMESTEPS x BATCHSIZE x HIDDEN_SIZE)
        # h_t: (DIRECTIONS x BATCHSIZE x HIDDEN_SIZE)  2 for bidirectional, 1 otherwise
        # c_t: (DIRECTIONS x BATCHSIZE x HIDDEN_SIZE)  2 for bidirectional, 1 otherwise
        out, (h_t, c_t) = self.rnn(x)

        # ?? what to do with lengths here
        # unpack sequence for use in linear layer
        unpacked_out, lengths_out = pad_packed_sequence(out, batch_first=False)

        # out: (N_TIMESTEPS x BATCHSIZE x N_LABELS)
        linear_out = self.linear(unpacked_out)

        out = nn.functional.log_softmax(linear_out, dim=2)

        # logging.info('--forward--')
        # logging.info(f'x_t:{unpacked_x.shape},lengths_x:{lengths_x}')
        # logging.info(f'h_t:{h_t.shape}', )
        # logging.info(f'c_t:{c_t.shape}')
        # logging.info(f'out_t:{unpacked_out.shape}, lengths_out:{lengths_out}')
        # logging.info(f'linear:{linear_out.shape}')
        # logging.info(f'softmax:{out.shape}')
        # logging.info(f'linear:{linear_out[0][0]}')
        # logging.info(f'softmax:{out[0][0]}')
        # logging.info('')
        return out
