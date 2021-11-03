"""
All things related to LSTMs.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from pynvml import *
from collections import OrderedDict
import numpy as np


# source for orthogonal initializer
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
# with minor changes
def sample(shape, gain):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return np.asarray(gain * q, dtype=np.float64)


def initialize_lstm_weights(shape):
    gain = np.sqrt(2)  # relu (default gain = 1.0)
    w = sample(shape, gain)
    return torch.FloatTensor(w)


def check_status():
    # check gpu properties
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    logging.info(f'total_memory:{t}')
    logging.info(f'free inside reserved:{f}')

    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    logging.info(f'total    : {info.total}')
    logging.info(f'free     : {info.free}')
    logging.info(f'used     : {info.used}')


class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, bidirectional, dropout):
        super(BasicLSTM, self).__init__()

        # TODO: cnn could go here (40 -> 128, 256 output channels) feature extractor
        # TODO: kaiming initialization for cnn
        # TODO: batch normalization for cnn

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False,  # faster if we don't do batch first
                            dropout=dropout,
                            bidirectional=bidirectional)

        direction = 2 if bidirectional else 1
        self.linear = nn.Linear(hidden_size * direction, output_size)  # ??use bias=False?
        # TODO: xavier initialization for linear
        # TODO: could do batch normalization after linear

    def forward(self, x, lengths_x):
        # pack sequence after any cnn layers
        # x: (N_TIMESTEPS x BATCHSIZE x FEATURES)
        packed_x = pack_padded_sequence(x, lengths_x.cpu(), enforce_sorted=True)

        # TODO: initialize hidden layer and cell state layer -  look into this
        # out: (N_TIMESTEPS x BATCHSIZE x HIDDEN_SIZE * DIRECTIONS)
        # h_t: (DIRECTIONS x BATCHSIZE x HIDDEN_SIZE)  DIRECTIONS=2 for bidirectional, 1 otherwise
        # c_t: (DIRECTIONS x BATCHSIZE x HIDDEN_SIZE)  DIRECTIONS=2 for bidirectional, 1 otherwise
        out, (h_t, c_t) = self.lstm(packed_x)

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


def _calc_output_size_from_dict(input_size, parm_dict):
    """
    Calculate the output size given a dictionary of parameters.
    Args:
        input_size (int): size of the input
        parm_dict (Dict): dictionary containing padding,dilation,kernel_size,and stride

    Returns:int representing output size

    """
    padding = parm_dict['padding']
    dilation = parm_dict['dilation']
    kernel_size = parm_dict['kernel_size']
    stride = parm_dict['stride']

    output_size = _calc_output_size(input_size, padding, dilation, kernel_size, stride)
    return output_size


def _init_weights(layer):
    """
    Perform initialization of layer weights if layer is a Conv2d layer.
    Args:
        layer: layer under consideration

    Returns: None

    """
    if isinstance(layer, nn.Conv1d):
        logging.info('initializing conv layer with kaiming normal...')
        nn.init.kaiming_normal_(layer.weight)
    elif isinstance(layer, nn.Linear):
        logging.info('initializing linear layer with xavier uniform...')
        nn.init.xavier_uniform_(layer.weight)
    else:
        logging.info('not initializing layer...')


# # build cnn layers
# sequence = []
# for i, conv_dict in enumerate(conv_dicts):
#     # convolution
#     layer_name = 'conv' + str(i + 1)
#     conv_tuple = (layer_name, nn.Conv1d(**conv_dict))
#     sequence.append(conv_tuple)
#
# self.cnn_layers = nn.Sequential(OrderedDict(sequence))

# self.cnn = nn.Conv1d(**conv_dict)
# nn.init.kaiming_normal_(self.cnn.weight)


# # initialize weights
# self.cnn_layers.apply(_init_weights)
#        sequence = []
# layer_name = 'lin1'
# lin_tuple = (layer_name, nn.Linear(hidden_size * direction, lin1_output_size))
# sequence.append(lin_tuple)

# layer_name = 'bn1'
# bn_tuple = (layer_name, nn.BatchNorm1d(lin1_output_size))
# sequence.append(bn_tuple)
#
# # dropout (optional)
# if lin1_dropout > 0:
#     layer_name = 'drop1'
#     drop_tuple = (layer_name, nn.Dropout(lin1_dropout))
#     sequence.append(drop_tuple)

# layer_name = 'relu1'
# relu_tuple = (layer_name, nn.ReLU(inplace=True))
# sequence.append(relu_tuple)

# layer_name = 'lin2'
# lin_tuple = (layer_name, nn.Linear(lin1_output_size, output_size))
# sequence.append(lin_tuple)
# self.linear_layers = nn.Sequential(OrderedDict(sequence))
# self.linear_layers.apply(_init_weights)
# self.linear = nn.Linear(hidden_size * direction, output_size)  # ??use bias=False?
# nn.init.xavier_uniform_(self.linear.weight)  # ?? correct choice
# TODO: could do batch normalization after linear

class CnnLSTM(nn.Module):
    def __init__(self, lstm_input_size, hidden_size, num_layers,
                 output_size, bidirectional, dropout, conv_dicts, lin1_output_size, lin1_dropout):
        super(CnnLSTM, self).__init__()

        self.cnn1 = nn.Conv1d(**conv_dicts[0])
        self.cnn2 = nn.Conv1d(**conv_dicts[1])

        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False,  # faster if we don't do batch first
                            dropout=dropout,
                            bidirectional=bidirectional)

        # build linear layers
        direction = 2 if bidirectional else 1

        # linear
        self.lin1 = nn.Linear(hidden_size * direction, lin1_output_size)

        # activation
        self.relu1 = nn.ReLU(inplace=True)

        # linear
        self.lin2 = nn.Linear(lin1_output_size, output_size)

    def forward(self, x, lengths_x, i):
        # X: (N_TIMESTEPS x BATCHSIZE x FEATURES)
        batchsize = x.shape[1]
        if i == 0:
            logging.info(f'x:{x.shape}')

        # transpose to shape expected for cnn
        x_transposed1 = torch.transpose(x, 0, 1)  # (BATCHSIZE x N_TIMESTEPS x FEATURES)
        if i == 0:
            logging.info(f'x_transposed1:{x_transposed1.shape}')

        x_transposed2 = torch.transpose(x_transposed1, 1, 2)  # (BATCHSIZE x FEATURES x N_TIMESTEPS) ??is this correct
        if i == 0:
            logging.info(f'x_transposed2:{x_transposed2.shape}')

        out_cnn1 = self.cnn1(x_transposed2)  # (BATCHSIZE x OUT_CHANNELS x N_TIMESTEPS)
        if i == 0:
            logging.info(f'out_cnn1:{out_cnn1.shape}')

        out_cnn2 = self.cnn2(out_cnn1)  # (BATCHSIZE x OUT_CHANNELS x N_TIMESTEPS)
        if i == 0:
            logging.info(f'out_cnn2:{out_cnn2.shape}')

        # transpose to match expectations for lstm
        x_transposed3 = torch.transpose(out_cnn2, 0, 1)  # (OUT_CHANNELS x BATCHSIZE x N_TIMESTEPS)
        if i == 0:
            logging.info(f'x_transposed3:{x_transposed3.shape}')

        x_transposed4 = torch.transpose(x_transposed3, 0, 2)  # (N_TIMESTEPS x BATCHSIZE x OUT_CHANNELS)
        if i == 0:
            logging.info(f'x_transposed4:{x_transposed4.shape}')

        # pack sequence after any cnn layers
        # packed_x: (N_TIMESTEPS x BATCHSIZE x OUT_CHANNELS)
        x_packed = pack_padded_sequence(x_transposed4, lengths_x.cpu(), enforce_sorted=True)

        # initialize hidden layers in LSTM
        directions = 2 if self.lstm.bidirectional else 1
        num_layers = self.lstm.num_layers
        hidden_size = self.lstm.hidden_size
        h_0 = initialize_lstm_weights((directions * num_layers, batchsize, hidden_size))
        c_0 = initialize_lstm_weights((directions * num_layers, batchsize, hidden_size))
        if i == 0:
            logging.info(f'h_0:{h_0.shape}')

        if i == 0:
            logging.info(f'c_0:{c_0.shape}')

        # move initial tensors to gpu if necessary
        if i == 0:
            logging.info(f'device for model is:{next(self.parameters()).device}')
        if 'cuda' in str(next(self.parameters()).device):
            if i == 0:
                logging.info('move to cuda')
            h_0 = h_0.to(device=torch.device('cuda'))
            c_0 = c_0.to(device=torch.device('cuda'))

        # out: (N_TIMESTEPS x BATCHSIZE x HIDDEN_SIZE * DIRECTIONS)
        # h_t: (DIRECTIONS x BATCHSIZE x HIDDEN_SIZE)  DIRECTIONS=2 for bidirectional, 1 otherwise
        # c_t: (DIRECTIONS x BATCHSIZE x HIDDEN_SIZE)  DIRECTIONS=2 for bidirectional, 1 otherwise
        out_lstm, (h_t, c_t) = self.lstm(x_packed, (h_0, c_0))

        # unpack sequence for use in linear layer
        # unpacked_x: (N_TIMESTEPS x BATCHSIZE x HIDDEN_SIZE * DIRECTIONS)
        # lengths_x: (BATCHSIZE,)
        x_unpacked, lengths_out = pad_packed_sequence(out_lstm, batch_first=False)
        if i == 0:
            logging.info(f'x_unpacked:{x_unpacked.shape}')

        x_transposed5 = torch.transpose(x_unpacked, 0, 1)  # (BATCHSIZE x N_TIMESTEPS x HIDDEN_SIZE * DIRECTIONS)
        if i == 0:
            logging.info(f'x_transposed5:{x_transposed5.shape}')

        x_linear1 = self.lin1(x_transposed5)  # (BATCHSIZE x N_TIMESTEPS x OUTPUTSIZE)
        if i == 0:
            logging.info(f'x_linear1:{x_linear1.shape}')

        x_relu1 = self.relu1(x_linear1)
        if i == 0:
            logging.info(f'x_relu1:{x_relu1.shape}')

        x_linear2 = self.lin2(x_relu1)  # x: ( BATCHSIZE x N_TIMESTEPS x N_LABELS)
        if i == 0:
            logging.info(f'x_linear2:{x_linear2.shape}')

        # dim=2 because N_LABELS is the dimension to softmax
        out_softmax = nn.functional.log_softmax(x_linear2, dim=2)
        if i == 0:
            logging.info(f'out_softmax:{out_softmax.shape}')

        x_transposed6 = torch.transpose(out_softmax, 0, 1)  # (N_TIMESTEPS x BATCHSIZE x N_LABELS)
        if i == 0:
            logging.info(f'x_transposed6:{x_transposed6.shape}')

        # transpose back to expected shape
        return x_transposed6
