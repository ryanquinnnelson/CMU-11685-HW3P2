"""
Handler for LSTM models.
"""
__author__ = 'ryanquinnnelson'

import logging

from octopus.models import LSTM


class LstmHandler:
    def __init__(self, model_type, input_size, hidden_size, num_layers,
                 output_size, bidirectional, dropout):
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        logging.info('Initializing LSTM handler...')

    def get_model(self):
        model = None

        if self.model_type == 'BasicLSTM':
            model = LSTM.BasicLSTM(self.input_size, self.hidden_size, self.num_layers,
                                   self.output_size, self.bidirectional, self.dropout)
        return model
