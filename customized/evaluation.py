"""
Evaluation phase customized to this dataset.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch
import numpy as np


def _convert_output(out):
    """
    Convert 2D output to 1D a single class label.

    Args:
        out (np.array): 2D output in which each row is a datapoint and each column is a single class

    Returns: np.array 1D output

    """
    out = np.argmax(out, axis=1)  # column with max value in each row is the index of the predicted label

    return out


def _calculate_num_hits(out, actual):
    """
    Calculate the number of accurate labels, using the desired (actual) labels.
    Args:
        out (torch.FloatTensor) : 2D tensor representing model output
        actual (torch.LongTensor): 1D tensor representing desired class labels

    Returns: int representing number of accurate labels

    """
    # retrieve labels from device by converting to numpy arrays
    actual = actual.cpu().detach().numpy()

    # convert output to class labels
    pred = _convert_output(out)

    # compare predictions against actual
    n_hits = np.sum(pred == actual)

    return n_hits


class Evaluation:
    """
    Defines an object to manage the evaluation phase of training.
    """

    def __init__(self, val_loader, criterion_func, devicehandler):
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
        num_hits = 0

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for i, (inputs, targets) in enumerate(self.val_loader):
                # prep
                inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

                # forward pass
                out = model.forward(inputs)

                # calculate validation loss
                loss = self.criterion_func(out, targets)
                val_loss += loss.item()

                # calculate number of accurate predictions for this batch
                out = out.cpu().detach().numpy()  # extract from gpu
                num_hits += _calculate_num_hits(out, targets)

                # delete mini-batch from device
                del inputs
                del targets

            # calculate evaluation metrics
            val_loss /= len(self.val_loader)  # average per mini-batch
            val_acc = num_hits / len(self.val_loader.dataset)

            return val_loss, val_acc


class EvaluationCenterLoss:
    """
    Defines an object to manage the evaluation phase of training in the case of using centerloss.
    """

    def __init__(self, val_loader, label_criterion_func, centerloss_func, centerloss_weight, devicehandler):
        """
        Initialize Evaluation object.

        Args:
            val_loader (DataLoader): DataLoader for validation dataset
            label_criterion_func (class): loss function for labels
            centerloss_func (class): loss function for centerloss
            centerloss_weight (float): importance of centerloss in overall loss calculation
            devicehandler (DeviceHandler): object to manage interaction of model/data and device
        """
        logging.info('Loading evaluation phase...')
        self.val_loader = val_loader
        self.label_criterion_func = label_criterion_func
        self.centerloss_func = centerloss_func
        self.centerloss_weight = centerloss_weight
        self.devicehandler = devicehandler

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
        num_hits = 0

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for i, (inputs, targets) in enumerate(self.val_loader):
                # prep
                inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

                # forward pass
                feature, outputs = model.forward(inputs, return_embedding=True)

                # calculate validation loss
                l_loss = self.label_criterion_func(outputs, targets)
                c_loss = self.centerloss_func(feature, targets)
                loss = l_loss + c_loss * self.centerloss_weight
                val_loss += loss.item()

                # calculate number of accurate predictions for this batch
                outputs = outputs.cpu().detach().numpy()  # extract from gpu
                num_hits += _calculate_num_hits(outputs, targets)

                # delete mini-batch from device
                del inputs
                del targets

            # calculate evaluation metrics
            val_loss /= len(self.val_loader)  # average per mini-batch
            val_acc = num_hits / len(self.val_loader.dataset)

            return val_loss, val_acc
