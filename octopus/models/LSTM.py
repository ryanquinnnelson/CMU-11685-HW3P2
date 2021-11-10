"""
All things related to LSTMs.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from pynvml import *
import numpy as np


# source for orthogonal initializer
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
# with minor changes
def sample(shape, gain):
    """
    Generate an orthogonal random array with given shape and gain.

    :param shape (Tuple): defines the shape of the array
    :param gain (float): Value to scale orthogonal values of array by.
    :return: np.array
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)

    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return np.asarray(gain * q, dtype=np.float64)


def initialize_lstm_weight(shape):
    """
    Initialize an orthogonal random array with given shape.
    :param shape (Tuple): defines the shape of the array
    :return: np.array
    """
    gain = np.sqrt(2)  # relu (default gain = 1.0)
    w = sample(shape, gain)
    return torch.FloatTensor(w)


class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, bidirectional, dropout):
        super(BasicLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False,  # faster if we don't do batch first
                            dropout=dropout,
                            bidirectional=bidirectional)

        direction = 2 if bidirectional else 1
        self.linear = nn.Linear(hidden_size * direction, output_size)  # ??use bias=False?

    def forward(self, x, lengths_x):
        # pack sequence after any cnn layers
        packed_x = pack_padded_sequence(x, lengths_x.cpu(), enforce_sorted=True)

        # lstm
        out, (h_t, c_t) = self.lstm(packed_x)

        # unpack sequence for use in linear layer
        unpacked_out, lengths_out = pad_packed_sequence(out, batch_first=False)

        # linear
        linear_out = self.linear(unpacked_out)

        # log softmax
        out = nn.functional.log_softmax(linear_out, dim=2)

        return out


def _calc_output_size(input_size, padding, dilation, kernel_size, stride):
    """
    Calculate the output size based on all parameters.

    Args:
        input_size (int): size of the input
        padding (int): amount of padding
        dilation (int): amount of dilation
        kernel_size (int): size of the kernel
        stride (int): size of the stride

    Returns: int representing output size

    """
    input_size_padded = input_size + 2 * padding
    kernel_dilated = (kernel_size - 1) * (dilation - 1) + kernel_size
    output_size = (input_size_padded - kernel_dilated) // stride + 1
    return output_size


def initialize_lstm_weights(bidirectional, num_layers, hidden_size, batchsize, device):
    """
    Initialize the hidden and cell state layers of the LSTM to be orthogonal matrices with a gain of sqrt(2).

    :param bidirectional (Boolean): True if LSTM is bidirectional. False otherwise.
    :param num_layers (int): Number of LSTM layers in LSTM portion of the model.
    :param hidden_size (int): Dimension of each hidden layer in the LSTM model.
    :param batchsize (int): Size of the batch.
    :param device (str): Device data is being placed onto.
    :return:
    """
    directions = 2 if bidirectional else 1
    num_layers = num_layers
    hidden_size = hidden_size

    # generate initializations
    h_0 = initialize_lstm_weight((directions * num_layers, batchsize, hidden_size))
    c_0 = initialize_lstm_weight((directions * num_layers, batchsize, hidden_size))

    # move initial tensors to gpu if necessary
    if 'cuda' in str(device):
        h_0 = h_0.to(device=torch.device('cuda'))
        c_0 = c_0.to(device=torch.device('cuda'))

    return h_0, c_0


class CnnLSTM(nn.Module):
    def __init__(self, lstm_input_size, hidden_size, num_layers,
                 output_size, bidirectional, lstm_dropout, conv_dicts, linear1_output_size, linear1_dropout):
        """
        Initialize CnnLSTM model.

        :param lstm_input_size (int): Dimension of features being input into the LSTM portion of the model.
        :param hidden_size (int): Dimension of each hidden layer in the LSTM model.
        :param num_layers (int): Number of LSTM layers in LSTM portion of the model.
        :param output_size (int): The number of labels in the feature dimension of linear layer output.
        :param bidirectional (Boolean): True if LSTM is bidirectional. False otherwise.
        :param lstm_dropout (float): The percent of node dropout in the LSTM model.
        :param conv_dicts (List): A List of dictionaries where each dictionary contains the parameters for a single cnn layer.
        :param lin1_output_size (int): The number of labels in the feature dimension of the first linear layer if there are multiple linear layers.
        :param lin1_dropout (float): The percent of node dropout in between linear layers in the model if there are multiple linear layers.
        """
        super(CnnLSTM, self).__init__()

        # cnn layers
        self.cnn1 = nn.Conv1d(**conv_dicts[0])
        nn.init.kaiming_normal_(self.cnn1.weight)

        self.bn1 = nn.BatchNorm1d(conv_dicts[0]['out_channels'])
        self.relu1 = nn.ReLU(inplace=True)

        self.cnn2 = nn.Conv1d(**conv_dicts[1])
        nn.init.kaiming_normal_(self.cnn2.weight)

        self.bn2 = nn.BatchNorm1d(conv_dicts[1]['out_channels'])
        self.relu2 = nn.ReLU(inplace=True)

        # lstm layers
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                            bidirectional=bidirectional)

        # linear layers
        direction = 2 if bidirectional else 1
        self.lin1 = nn.Linear(hidden_size * direction, linear1_output_size)
        nn.init.xavier_uniform_(self.lin1.weight)

        # additional linear layers are turned off due to poor performance
        # if linear1_dropout > 0:
        #     self.drop1 = nn.Dropout(linear1_dropout)
        # else:
        #     self.drop1 = None
        #
        # self.relu3 = nn.ReLU(inplace=True)

        # self.lin2 = nn.Linear(linear1_output_size, output_size)
        # nn.init.xavier_uniform_(self.lin2.weight)

        # softmax layer
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, x, lengths_x, phase):
        """
        Perform forward pass on model.

        :param x (Tensor): Data for this batch. Shape is (BATCHSIZE,N_TIMESTEPS,FEATURES)
        :param lengths_x (Tensor): Lengths of utterances for each record in this batch. Shape is (BATCHSIZE,UTTERANCE_LENGTH)
        :return: Tuple (Tensor,Tensor) representing (output,output_lengths). Model output has shape (N_TIMESTEPS,BATCHSIZE,N_LABELS). Output lengths are lengths of utterances for each record in this batch. Shape is (BATCHSIZE,UTTERANCE_LENGTH)
        """
        # save batch size for later
        batchsize = x.shape[0]

        # transpose to shape expected for cnn
        x_transposed1 = torch.transpose(x, 1, 2)  # (BATCHSIZE,FEATURES,N_TIMESTEPS)

        # cnn
        out_cnn1 = self.cnn1(x_transposed1)  # (BATCHSIZE,OUT_CHANNELS1,N_TIMESTEPS)
        out_cnn1 = self.bn1(out_cnn1)
        out_cnn1 = self.relu1(out_cnn1)
        out_cnn2 = self.cnn2(out_cnn1)  # (BATCHSIZE,OUT_CHANNELS2,N_TIMESTEPS)
        out_cnn2 = self.bn2(out_cnn2)
        out_cnn2 = self.relu2(out_cnn2)

        # transpose to match shape requirements for lstm
        x_transposed2 = torch.transpose(out_cnn2, 1, 2)  # (BATCHSIZE,N_TIMESTEPS,OUT_CHANNELS2)

        # pack sequence after any cnn layers
        if phase == 'testing':
            x_packed = pack_padded_sequence(x_transposed2,  # (BATCHSIZE,N_TIMESTEPS,OUT_CHANNELS2)
                                            lengths_x.cpu(),
                                            enforce_sorted=False,  # test batch isn't sorted
                                            batch_first=True)
        else:
            x_packed = pack_padded_sequence(x_transposed2,  # (BATCHSIZE,N_TIMESTEPS,OUT_CHANNELS2)
                                            lengths_x.cpu(),
                                            enforce_sorted=True,
                                            batch_first=True)

        # initialize hidden layers in LSTM
        h_0, c_0 = initialize_lstm_weights(self.lstm.bidirectional, self.lstm.num_layers, self.lstm.hidden_size,
                                           batchsize, next(self.parameters()).device)

        # out_lstm: (BATCHSIZE,N_TIMESTEPS,HIDDEN_SIZE * DIRECTIONS)
        # h_t: (DIRECTIONS * NUM_LAYERS,BATCHSIZE,HIDDEN_SIZE)  DIRECTIONS=2 for bidirectional, 1 otherwise
        # c_t: (DIRECTIONS * NUM_LAYERS,BATCHSIZE,HIDDEN_SIZE)  DIRECTIONS=2 for bidirectional, 1 otherwise
        out_lstm, (h_t, c_t) = self.lstm(x_packed, (h_0, c_0))

        # unpack sequence for use in linear layer
        # unpacked_x: (BATCHSIZE,N_TIMESTEPS,HIDDEN_SIZE * DIRECTIONS)
        # lengths_x: (BATCHSIZE,)
        x_unpacked, lengths_out = pad_packed_sequence(out_lstm, batch_first=True)

        # linear
        x_linear1 = self.lin1(x_unpacked)  # (BATCHSIZE,N_TIMESTEPS,LINEAR1_OUTPUT_SIZE)

        # additional linear layers are turned off due to poor performance
        # # optional dropout layer
        # if self.drop1 is not None:
        #     x_linear1 = self.drop1(x_linear1)  # (BATCHSIZE,N_TIMESTEPS,LINEAR1_OUTPUT_SIZE)
        #
        # x_relu3 = self.relu3(x_linear1)  # (BATCHSIZE,N_TIMESTEPS,LINEAR1_OUTPUT_SIZE)
        # x_linear2 = self.lin2(x_relu3)  # (BATCHSIZE,N_TIMESTEPS,N_LABELS)

        # log softmax
        out_softmax = self.logsoftmax(x_linear1)  # (BATCHSIZE,N_TIMESTEPS,N_LABELS)

        # transpose to expected shape for CTCLoss # (expects batch second for input)
        x_transposed3 = torch.transpose(out_softmax, 0, 1)  # (N_TIMESTEPS,BATCHSIZE,N_LABELS)

        return x_transposed3, lengths_out
