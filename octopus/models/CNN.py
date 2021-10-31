"""
Defines all standard CNNs octopus can generate.
"""

__author__ = 'ryanquinnnelson'

import logging

import torch.nn as nn
from collections import OrderedDict

from octopus.models.helper import _get_activation_function


def _get_pool_class(pool_class):
    """
    Obtain the pooling object given the parameter.

    Args:
        pool_class (str): represents the pooling layer to use

    Returns: nn pooling layer

    """
    pool = None

    if pool_class == 'MaxPool2d':
        pool = nn.MaxPool2d

    return pool


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


def _build_cnn2d_sequence(input_size, activation_func, batch_norm, conv_dicts, pool_class, pool_dicts):
    """
    Construct a List of all layers in the CNN.

    Args:
        input_size (int): size of the input
        activation_func (str): represents the activation function to use after each cnn layer
        batch_norm (boolean): True if batch normalization should be used after each cnn layer
        conv_dicts (Dict): dictionary containing the parameters for each cnn layer
        pool_class (str): represents the pooling class to use for each pooling layer
        pool_dicts (Dict): dictionary containing the parameters for each pooling layer

    Returns: List of all layers in the CNN

    """
    sequence = []

    # track sizes after each layer
    layer_input_size = input_size
    layer_output_size = None
    out_channels = None
    for i, conv_dict in enumerate(conv_dicts):  # create a layer for each parameter dictionary
        # print('start' + str(i + 1), layer_input_size, layer_output_size)

        # convolution
        layer_name = 'conv' + str(i + 1)
        conv_tuple = (layer_name, nn.Conv2d(**conv_dict))
        sequence.append(conv_tuple)
        layer_output_size = _calc_output_size_from_dict(layer_input_size, conv_dict)
        out_channels = conv_dict['out_channels']
        # print('conv' + str(i + 1), layer_input_size, layer_output_size)
        logging.info(f'{layer_name}:[{layer_input_size},{layer_output_size}]')

        # batch normalization
        if batch_norm:
            layer_name = 'bn' + str(i + 1)
            bn_tuple = (layer_name, nn.BatchNorm2d(num_features=conv_dict['out_channels']))
            sequence.append(bn_tuple)

        # activation layer
        layer_name = activation_func + str(i + 1)
        activation_tuple = (layer_name, _get_activation_function(activation_func))
        sequence.append(activation_tuple)

        # pooling layer
        if len(pool_dicts) > i:
            pool_dict = pool_dicts[i]
            layer_name = 'pool' + str(i + 1)
            layer_pool_class = _get_pool_class(pool_class)
            pool_tuple = (layer_name, layer_pool_class(**pool_dict))
            sequence.append(pool_tuple)

            # update input and output sizes based on pooling layer
            layer_input_size = layer_output_size
            layer_output_size = _calc_output_size_from_dict(layer_input_size, pool_dict)
            logging.info(f'{layer_name}:[{layer_input_size},{layer_output_size}]')
            # print('pool' + str(i + 1), layer_input_size, layer_output_size)
            layer_input_size = layer_output_size  # becomes layer_input_size for next cnn layer
            # print('pool' + str(i + 1), layer_input_size, layer_output_size)

    return sequence, layer_output_size, out_channels


# TODO - add desired number of linear layers
def _build_linear_sequence(input_size, output_size, out_channels):
    """
    Construct a List of the linear layers in the CNN.
    Args:
        input_size (int): Size of the output expected from the last CNN layer
        output_size (int): Size of the desired output layer
        out_channels (int): Number of output channels expected from the last CNN layer

    Returns: List of all linear layers in the CNN

    """
    # add flattening as first layer
    sequence = [('flat', nn.Flatten())]

    # add one or more linear layers
    sequence.append(('lin', nn.Linear(input_size * input_size * out_channels, output_size)))

    # add softmax as final activation
    sequence.append(('soft', nn.Softmax(dim=1)))  # all rows sum to 1
    return sequence


class CNN2d(nn.Module):
    """
    Defines a generic CNN model.
    """
    def __init__(self, input_size, output_size, activation_func, batch_norm, conv_dicts, pool_class, pool_dicts):
        """
        Initialize a CNN2d object.

        Args:
            input_size (int): size of the input
            output_size (int): Size of the desired output layer
            activation_func (str): represents the activation function to use after each cnn layer
            batch_norm (boolean): True if batch normalization should be used after each cnn layer
            conv_dicts (Dict): dictionary containing the parameters for each cnn layer
            pool_class (str): represents the pooling class to use for each pooling layer
            pool_dicts (Dict): dictionary containing the parameters for each pooling layer
        """
        super(CNN2d, self).__init__()

        # define cnn layers
        cnn_sequence, cnn_output_size, cnn_out_channels = _build_cnn2d_sequence(input_size, activation_func, batch_norm,
                                                                                conv_dicts,
                                                                                pool_class, pool_dicts)
        self.cnn_layers = nn.Sequential(OrderedDict(cnn_sequence))

        # define linear layers
        linear_sequence = _build_linear_sequence(cnn_output_size, output_size, cnn_out_channels)
        self.linear_layers = nn.Sequential(OrderedDict(linear_sequence))

    def forward(self, x):
        """
        Execute a forward pass on the model.
        Args:
            x (Tensor): data for this batch

        Returns: Tensor of model output

        """
        x = self.cnn_layers(x)
        x = self.linear_layers(x)
        return x
