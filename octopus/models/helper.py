"""
Helper for models.
"""
__author__ = 'ryanquinnnelson'

from torch import nn as nn


def _get_activation_function(activation_func):
    """
    Obtain the loss function based on the given argument.

    Args:
        activation_func (str): represents loss function to use

    Returns: loss function

    """
    act = None

    if activation_func == 'ReLU':
        act = nn.ReLU(inplace=True)
    elif activation_func == 'LeakyReLU':
        act = nn.LeakyReLU(inplace=True)
    elif activation_func == 'Sigmoid':
        act = nn.Sigmoid()
    elif activation_func == 'Tanh':
        act = nn.Tanh()

    return act
