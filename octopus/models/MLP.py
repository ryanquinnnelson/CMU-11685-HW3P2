"""
Defines all standard MLP models that Octopus can generate.
"""

__author__ = 'ryanquinnnelson'

from collections import OrderedDict

import torch.nn as nn

from octopus.models.helper import _get_activation_function


def _build_linear_sequence(sizes, activation_func, dropout_rate, batch_norm):
    """
    Construct a List of all model layers in the MLP.
    Args:
        sizes (List): List of the size of each hidden layer, in layer order
        activation_func (str): activation function to use in each layer
        dropout_rate (float): Dropout rate
        batch_norm (Boolean): whether to use batch normalization after each linear layer

    Returns: List of all model layers in the MLP

    """
    sequence = []
    num_hidden_layers = len(sizes) - 2  # input and output not included
    for i in range(num_hidden_layers + 1):  # add final layer for output

        # linear layer
        layer_name = 'lin' + str(i + 1)
        linear_tuple = (layer_name, nn.Linear(sizes[i], sizes[i + 1]))
        sequence.append(linear_tuple)

        # add the following after each hidden linear layer
        if i < num_hidden_layers:  # not final layer

            # batch normalization layer
            if batch_norm:
                layer_name = 'bn' + str(i + 1)
                batch_norm_tuple = (layer_name, nn.BatchNorm1d(sizes[i + 1]))
                sequence.append(batch_norm_tuple)

            # dropout layer
            if dropout_rate > 0:
                layer_name = 'drop' + str(i + 1)
                dropout_tuple = (layer_name, nn.Dropout(dropout_rate))
                sequence.append(dropout_tuple)

            # activation layer
            layer_name = activation_func + str(i + 1)
            activation_tuple = (layer_name, _get_activation_function(activation_func))
            sequence.append(activation_tuple)

    return sequence


class MLP(nn.Module):
    """
    Defines an MLP model.

    """
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_layer_sizes,
                 activation_func,
                 dropout_rate,
                 batch_norm):
        """
        Initialize an MLP object.
        Args:
            input_size (int): size of the input
            output_size (int): size of the output
            hidden_layer_sizes (List): List of the size of each hidden layer, in layer order
            activation_func (str): activation function to use in each layer
            dropout_rate (float): Dropout rate
            batch_norm (Boolean): whether to use batch normalization after each linear layer
        """
        super(MLP, self).__init__()

        # set layer sizes
        sizes = [input_size] + hidden_layer_sizes + [output_size]

        # build layer tuples based on parameters
        sequence = _build_linear_sequence(sizes, activation_func, dropout_rate, batch_norm)

        # build Sequential model from sequence
        self.model = nn.Sequential(OrderedDict(sequence))

    def forward(self, x):
        """
        Perform forward pass on model.

        Args:
            x (Tensor): batch of data

        Returns: Tensor of model output

        """
        return self.model(x)  # self.layers(x) also works
