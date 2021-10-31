"""
All things related to models.
"""
__author__ = 'ryanquinnnelson'

import logging

from octopus.models.CNN import CNN2d


def _convert_dict_to_dicts(d):
    """
    Convert given dictionary to a list of dictionaries in which each dictionary in the list contains only the arguments
    related to a single CNN layer.

    Args:
        d (Dict): dictionary in which each key is composed of an integer representing a layer number and a parameter

    Returns: List of dictionaries

    """
    # determine number of dictionaries to create
    layer_number_set = set()
    for key in d:
        layer_number, layer_parm = key.strip().split('.')
        layer_number_set.add(int(layer_number))
    num_layers = len(layer_number_set)

    # create a dictionary of parmeters for each layer, in layer order (1,2,3...)
    layer_dicts = []
    for i in range(1, num_layers + 1):  # extract values in order

        layer_dict = {}

        # find all dictionary entries that start with this layer name
        # extract parameter names and values and place into a dictionary for this layer
        for key in d:
            layer_number, layer_parm = key.strip().split('.')
            if int(layer_number) == i:
                layer_dict[layer_parm] = d[key]
        layer_dicts.append(layer_dict)

    return layer_dicts


class CnnHandler:
    """
    Defines an object to handle constructing generic CNN models.
    """
    def __init__(self,
                 model_type, input_size, output_size, activation_func, batch_norm, conv_dict, pool_class, pool_dict):
        """

        Args:
            model_type (str): represents the type of CNN to construct
            input_size (int): size of the input
            output_size (int): Size of the desired output layer
            activation_func (str): represents the activation function to use after each cnn layer
            batch_norm (boolean): True if batch normalization should be used after each cnn layer
            conv_dicts (Dict): dictionary containing the parameters for each cnn layer
            pool_class (str): represents the pooling class to use for each pooling layer
            pool_dicts (Dict): dictionary containing the parameters for each pooling layer
        """
        logging.info('Initializing model handling...')
        self.model_type = model_type
        self.input_size = input_size
        self.output_size = output_size
        self.activation_func = activation_func
        self.batch_norm = batch_norm
        self.conv_dicts = _convert_dict_to_dicts(conv_dict)
        self.pool_class = pool_class
        self.pool_dicts = _convert_dict_to_dicts(pool_dict)

    def get_model(self):
        """
        Initialize the model using all parameters.
        Returns: nn.Module model

        """
        logging.info('Initializing model...')
        model = None

        if self.model_type == 'CNN2d':
            model = CNN2d(self.input_size, self.output_size, self.activation_func, self.batch_norm, self.conv_dicts,
                          self.pool_class, self.pool_dicts)

        logging.info(f'Model initialized:\n{model}')
        return model
