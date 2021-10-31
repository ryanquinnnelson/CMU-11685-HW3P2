"""
All things related to LSTMs.
"""
__author__ = 'ryanquinnnelson'

import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


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
        print('--forward--')
        unpacked_x, lengths_x = pad_packed_sequence(x, batch_first=False)
        print('x_t', unpacked_x.shape, 'lengths_x', lengths_x)

        # initialize hidden layer and cell state layer?
        # what was lengths used for here? was it returned in the dataloader too as a third arg?

        out, (h_t, c_t) = self.rnn(x)
        print('h_t', h_t.shape)
        print('c_t', c_t.shape)

        # unpack sequence for use in linear layer
        unpacked_out, lengths_out = pad_packed_sequence(out, batch_first=False)
        print('out_t', unpacked_out.shape, 'lengths_out', lengths_out)

        out = self.linear(unpacked_out)
        print('linear', out.shape)
        print()
        return out
