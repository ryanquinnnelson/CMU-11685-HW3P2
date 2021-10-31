"""
Evaluation phase customized to this dataset.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch
import numpy as np

import phoneme_list as pl
from customized.helper import convert_to_phonemes, target_to_phonemes, convert_to_string, decode_output


def calculate_distances(beam_results, out_lens, targets):
    import Levenshtein
    n_batches = beam_results.shape[0]
    logging.info(f'Calcuating distance for {n_batches} entries...')
    distances = []

    for i in range(n_batches):
        logging.info(f'targets[{i}]', targets[i])
        out_converted = convert_to_phonemes(i, beam_results, out_lens, pl.PHONEME_LIST)
        target_converted = target_to_phonemes(targets[i], pl.PHONEME_LIST)
        logging.info('out_converted', out_converted)
        logging.info('target_converted', target_converted)

        out_str = convert_to_string(out_converted)
        target_str = convert_to_string(target_converted)
        logging.info('out_str', out_str)
        logging.info('target_str', target_str)

        distance = Levenshtein.distance(out_str, target_str)
        logging.info('distance', distance)
        distances.append(distance)

    return np.array(distances).sum()


class Evaluation:
    """
    Defines an object to manage the evaluation phase of training.
    """

    def __init__(self, val_loader, criterion_func, devicehandler, ctcdecodehandler):
        """
        Initialize Evaluation object.

        Args:
            val_loader (DataLoader): DataLoader for validation dataset
            criterion_func (class): loss function
            devicehandler (DeviceHandler): object to manage interaction of model/data and device
        """
        logging.info('Loading evaluation phase...')
        self.val_loader = val_loader
        self.criterion_func = criterion_func
        self.devicehandler = devicehandler
        self.ctcdecode = ctcdecodehandler.get_ctcdecoder()

    def evaluate_model(self, epoch, num_epochs, model):
        """
        Perform evaluation phase of training.

        Args:
            epoch (int): Epoch being trained
            num_epochs (int): Total number of epochs to be trained
            model (nn.Module): model being trained

        Returns: Tuple (float,float) representing (val_loss, val_metric)

        """
        logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')
        val_loss = 0
        running_distance = 0

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for i, (inputs, targets, input_lengths, target_lengths) in enumerate(self.val_loader):
                # prep
                inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

                # forward pass
                out = model.forward(inputs)

                # calculate validation loss
                print('--compute loss--')
                print('targets', targets.shape)
                print('input_lengths', input_lengths, type(input_lengths), input_lengths.shape)
                print('target_lengths', target_lengths, type(target_lengths), target_lengths.shape)
                print('out', out.shape)

                loss = self.criterion_func(out, targets, input_lengths, target_lengths)
                val_loss += loss.item()

                # calculate distance between actual and desired output
                out = out.cpu().detach()  # extract from gpu
                beam_results, beam_scores, timesteps, out_lens = decode_output(out, self.ctcdecode)
                distance = calculate_distances(beam_results, out_lens, targets.cpu().detach())
                running_distance += distance

                # delete mini-batch from device
                del inputs
                del targets

            # calculate evaluation metrics
            val_loss /= len(self.val_loader)  # average per mini-batch
            avg_distance = running_distance / len(self.val_loader.dataset)

            return val_loss, avg_distance
