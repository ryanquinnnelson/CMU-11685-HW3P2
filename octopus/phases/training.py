"""
Defines the training phase of model training.
"""
__author__ = 'ryanquinnnelson'

import logging
import torch


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

            # compute forward pass
            out = model.forward(inputs)

            # calculate loss
            print('--compute loss--')
            print('targets', targets.shape)
            print('input_lengths', input_lengths, type(input_lengths), input_lengths.shape)
            print('target_lengths', target_lengths, type(target_lengths), target_lengths.shape)
            print('out', out.shape)

            loss = self.criterion_func(out, targets, input_lengths, target_lengths)
            train_loss += loss.item()

            # compute backward pass
            loss.backward()

            # update model weights
            optimizer.step()

            # delete mini-batch data from device
            del inputs
            del targets

        # calculate average loss across all mini-batches
        train_loss /= len(self.train_loader)

        return train_loss

