"""
All things related to LSTMs.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from pynvml import *


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


class CnnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, bidirectional, dropout, conv_dict):
        super(CnnLSTM, self).__init__()

        self.cnn = nn.Conv1d(**conv_dict)
        nn.init.kaiming_normal_(self.cnn.weight)
        layer_output_size = _calc_output_size_from_dict(input_size, conv_dict)
        # TODO: batch normalization for cnn

        self.lstm = nn.LSTM(input_size=layer_output_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False,  # faster if we don't do batch first
                            dropout=dropout,
                            bidirectional=bidirectional)

        direction = 2 if bidirectional else 1
        self.linear = nn.Linear(hidden_size * direction, output_size)  # ??use bias=False?
        nn.init.xavier_normal_(self.linear.weight)  # ?? correct choice
        # TODO: could do batch normalization after linear

    def forward(self, x, lengths_x, i):
        if i == 0:
            logging.info('forward pass')
            check_status()
        # expects batch size first, channels next
        x = torch.transpose(x, 0, 1)  # (BATCHSIZE x N_TIMESTEPS x FEATURES)
        x = torch.transpose(x, 1, 2)  # (BATCHSIZE x FEATURES x N_TIMESTEPS) ??is this correct

        if i == 0:
            logging.info('after transposing')
            check_status()
        x = self.cnn(x)  # (BATCHSIZE x OUT_CHANNELS x N_TIMESTEPS)

        if i == 0:
            logging.info('after cnn')
            check_status()
        # transpose dimensions to match expectations for remaining layers
        x = torch.transpose(x, 0, 1)  # (OUT_CHANNELS x BATCHSIZE x N_TIMESTEPS)
        x = torch.transpose(x, 0, 2)  # (N_TIMESTEPS x BATCHSIZE x OUT_CHANNELS)

        if i == 0:
            logging.info('after transposing again')
            check_status()
        # pack sequence after any cnn layers
        # packed_x: (N_TIMESTEPS x BATCHSIZE x OUT_CHANNELS)
        x = pack_padded_sequence(x, lengths_x.cpu(), enforce_sorted=True)
        if i == 0:
            logging.info('after packing')
            check_status()
        # TODO: initialize hidden layer and cell state layer -  look into this
        # out: (N_TIMESTEPS x BATCHSIZE x HIDDEN_SIZE * DIRECTIONS)
        # h_t: (DIRECTIONS x BATCHSIZE x HIDDEN_SIZE)  DIRECTIONS=2 for bidirectional, 1 otherwise
        # c_t: (DIRECTIONS x BATCHSIZE x HIDDEN_SIZE)  DIRECTIONS=2 for bidirectional, 1 otherwise
        x, (h_t, c_t) = self.lstm(x)
        if i == 0:
            logging.info('after lstm')
            check_status()
        # unpack sequence for use in linear layer
        # unpacked_x: (N_TIMESTEPS x BATCHSIZE x HIDDEN_SIZE * DIRECTIONS)
        # lengths_x: (BATCHSIZE,)
        x, lengths_out = pad_packed_sequence(x, batch_first=False)
        if i == 0:
            logging.info('after unpacking')
            check_status()
        # out: (N_TIMESTEPS x BATCHSIZE x N_LABELS)
        x = self.linear(x)
        if i == 0:
            logging.info('after linear')
            check_status()
        x = nn.functional.log_softmax(x, dim=2)  # N_LABELS is the dimension to softmax
        if i == 0:
            logging.info('after softmax')
            check_status()
        # logging.info('--forward--')
        # logging.info(f'x_t:{x.shape},lengths_x:{lengths_x}')
        # logging.info(f'x_transposed:{x_transposed1.shape}')
        # logging.info(f'x_transposed:{x_transposed2.shape}')
        # logging.info(f'cnn_t:{out_cnn.shape}')
        # logging.info(f'out_cnn_transposed:{out_cnn_transposed1.shape}')
        # logging.info(f'out_cnn_transposed:{out_cnn_transposed2.shape}')
        # logging.info(f'out_t:{unpacked_out.shape}, lengths_out:{lengths_out}')
        # logging.info(f'h_t:{h_t.shape}')
        # logging.info(f'c_t:{c_t.shape}')
        # logging.info(f'linear:{linear_out.shape}')
        # logging.info(f'softmax:{out.shape}')
        # logging.info('')
        return x
