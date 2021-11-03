"""
All things related to training phases.
"""

import numpy as np

from customized.helper import decode_output, calculate_distances, check_status

import logging
import torch
import warnings
from customized.helper import out_to_phonemes, target_to_phonemes, convert_to_string, decode_output
import customized.phoneme_list as pl

warnings.filterwarnings('ignore')


class Training:
    """
    Defines object to manage Training phase of training.
    """

    def __init__(self, train_loader, criterion_func, devicehandler):
        """
        Initialize Training object.

        Args:
            train_loader (DataLoader): DataLoader for training data
            criterion_func (class): loss function
            devicehandler (DeviceHandler):manages device on which training is being run
        """
        logging.info('Loading training phase...')
        self.train_loader = train_loader
        self.criterion_func = criterion_func
        self.devicehandler = devicehandler

    def train_model(self, epoch, num_epochs, model, optimizer):
        """
        Executes one epoch of training.

        Args:
            epoch (int): Epoch being trained
            num_epochs (int): Total number of epochs to be trained
            model (nn.Module): model being trained
            optimizer (nn.optim): optimizer for this model

        Returns: float representing average training loss

        """
        logging.info(f'Running epoch {epoch}/{num_epochs} of training...')
        train_loss = 0

        # Set model in 'Training mode'
        model.train()

        # process mini-batches
        for i, (inputs, targets, input_lengths, target_lengths) in enumerate(self.train_loader):
            # prep
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)
            input_lengths, target_lengths = self.devicehandler.move_data_to_device(model, input_lengths, target_lengths)
            # if i == 0:
            #     logging.info('after loading data')
            #     check_status()

            # compute forward pass
            # inputs: (N_TIMESTEPS x BATCHSIZE x FEATURES)
            # out: (N_TIMESTEPS x BATCHSIZE x N_LABELS)
            out = model.forward(inputs, input_lengths, i)
            # if i == 0:
            #     logging.info('after forward pass')
            #     check_status()

            # calculate validation loss
            # targets: (N_TIMESTEPS x UTTERANCE_LABEL_LENGTH)
            loss = self.criterion_func(out, targets, input_lengths, target_lengths)
            train_loss += loss.item()
            # logging.info('--compute loss--')
            # logging.info(f'targets:{targets.shape}')
            # logging.info(f'input_lengths:{input_lengths},{input_lengths.shape}')
            # logging.info(f'target_lengths:{target_lengths},{target_lengths.shape}')
            # logging.info(f'loss:{loss.item()}')
            # logging.info('')

            # if i == 0:
            #     logging.info('after calculating loss')
            #     check_status()

            # delete mini-batch data from device
            del inputs
            del targets
            del input_lengths
            del target_lengths

            # if i == 0:
            #     logging.info('after deleting data')
            #     check_status()

            # compute backward pass
            loss.backward()

            # if i == 0:
            #     logging.info('after backward pass')
            #     check_status()

            # update model weights
            optimizer.step()

        # calculate average loss across all mini-batches
        train_loss /= len(self.train_loader)

        return train_loss


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
        self.ctcdecodehandler = ctcdecodehandler

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
        ctcdecode = self.ctcdecodehandler.ctcdecoder()
        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for i, (inputs, targets, input_lengths, target_lengths) in enumerate(self.val_loader):
                # prep
                inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)
                input_lengths, target_lengths = self.devicehandler.move_data_to_device(model, input_lengths,
                                                                                       target_lengths)

                # if i == 0:
                #     logging.info('after loading data')
                #     check_status()

                # compute forward pass
                # inputs: (N_TIMESTEPS x BATCHSIZE x FEATURES)
                # out: (N_TIMESTEPS x BATCHSIZE x N_LABELS)
                out = model.forward(inputs, input_lengths, i)
                # if i == 0:
                #     logging.info('after forward pass')
                #     check_status()

                # calculate validation loss
                # targets: (N_TIMESTEPS x UTTERANCE_LABEL_LENGTH)
                loss = self.criterion_func(out, targets, input_lengths, target_lengths)
                val_loss += loss.item()
                # logging.info('--compute loss--')
                # logging.info(f'targets:{targets.shape}')
                # logging.info(f'input_lengths:{input_lengths},{input_lengths.shape}')
                # logging.info(f'target_lengths:{target_lengths},{target_lengths.shape}')
                # logging.info(f'loss:{loss.item()}')
                # logging.info('')
                # if i == 0:
                #     logging.info('after calculating loss')
                #     check_status()

                # calculate distance between actual and desired output
                out = out.cpu().detach()  # extract from gpu
                # logging.info(f'out detached:{out.shape}')
                beam_results, beam_scores, timesteps, out_lens = decode_output(out, ctcdecode)
                distance = calculate_distances(beam_results, out_lens, targets.cpu().detach())
                running_distance += distance  # ?? something more for running total
                # if i == 0:
                #     logging.info('after calculating distances')
                #     check_status()

                # delete mini-batch from device
                del inputs
                del targets
                del target_lengths
                del input_lengths

                # if i == 0:
                #     logging.info('after deleting data')
                #     check_status()

            # calculate evaluation metrics
            val_loss /= len(self.val_loader)  # average per mini-batch
            avg_distance = running_distance / len(self.val_loader.dataset)

            return val_loss, avg_distance


class Testing:
    """
    Defines object to manage testing phase of training.
    """

    def __init__(self, test_loader, devicehandler, ctcdecodehandler):
        """
        Initialize Testing object.

        Args:
            test_loader (DataLoader): DataLoader for test data
            devicehandler (DeviceHandler): manages device on which training is being run
        """
        logging.info('Loading testing phase...')
        self.test_loader = test_loader
        self.devicehandler = devicehandler
        self.ctcdecodehandler = ctcdecodehandler

    def test_model(self, epoch, num_epochs, model):
        """
        Execute one epoch of model testing.

        Args:
            epoch (int): Epoch being trained
            num_epochs (int): Total number of epochs to be trained
            model (nn.Module): model being trained

        Returns: np.array of test output

        """
        logging.info(f'Running epoch {epoch}/{num_epochs} of testing...')
        output = []

        ctcdecode = self.ctcdecodehandler.ctcdecoder()

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for i, (inputs, input_lengths) in enumerate(self.test_loader):
                # prep
                inputs, targets = self.devicehandler.move_data_to_device(model, inputs, None)

                # compute forward pass
                # inputs: (N_TIMESTEPS x BATCHSIZE x FEATURES)
                # out: (N_TIMESTEPS x BATCHSIZE x N_LABELS)
                out = model.forward(inputs, input_lengths, i)

                # capture output for mini-batch
                out = out.cpu().detach()  # extract from gpu if necessary

                # decode output
                # out: (N_TIMESTEPS x BATCHSIZE x N_LABELS)
                beam_results, beam_scores, timesteps, out_lens = decode_output(out, ctcdecode)

                # convert to strings using phoneme map (not phoneme list)
                n = beam_results.shape[0]
                # logging.info(f'Converting {n_batches} beam results to phonemes...')

                for i in range(n):
                    out_converted = out_to_phonemes(i, beam_results, out_lens, pl.PHONEME_MAP)
                    # logging.info(f'out_converted[{i}]:{out_converted}')

                    converted_str = convert_to_string(out_converted)
                    # logging.info(f'converted_str[{i}]:{converted_str}')
                    output.append(converted_str)

        return output
