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
        for i, (inputs, targets) in enumerate(self.train_loader):
            # prep
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

            # compute forward pass
            out = model.forward(inputs)

            # calculate loss
            loss = self.criterion_func(out, targets)
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


class TrainingCenterLoss:
    """
    Defines object to manage training involving center loss.
    """

    def __init__(self, train_loader, label_criterion_func, centerloss_func, centerloss_weight, devicehandler):
        """

        Args:
            train_loader (DataLoader): DataLoader for training data
            label_criterion_func (class): loss function for the labels
            centerloss_func (class): loss function for centerloss
            centerloss_weight (float): importance of the centerloss in the overall loss calculation
            devicehandler (DeviceHandler): manages device on which training is being run
        """
        logging.info('Loading training phase...')
        self.train_loader = train_loader
        self.label_criterion_func = label_criterion_func
        self.centerloss_func = centerloss_func
        self.devicehandler = devicehandler
        self.centerloss_weight = centerloss_weight

    def train_model(self, epoch, num_epochs, model, optimizer_closs, optimizer_label):
        """
        Execute one epoch of model training under centerloss.

        Args:
            epoch (int): Epoch being trained
            num_epochs (int): Total number of epochs to be trained
            model (nn.Module): model being trained
            optimizer_closs (nn.optim): optimizer for the centerloss
            optimizer_label (nn.optim): optimizer for the labels

        Returns: float representing average training loss

        """
        logging.info(f'Running epoch {epoch}/{num_epochs} of training...')
        train_loss = 0

        # Set model in 'Training mode'
        model.train()

        # process mini-batches
        for i, (inputs, targets) in enumerate(self.train_loader):
            # prep
            optimizer_closs.zero_grad()
            optimizer_label.zero_grad()
            torch.cuda.empty_cache()
            inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

            # compute forward pass
            feature, outputs = model.forward(inputs, return_embedding=True)

            # calculate loss
            l_loss = self.label_criterion_func(outputs, targets)
            c_loss = self.centerloss_func(feature, targets)
            loss = l_loss + self.centerloss_weight * c_loss
            train_loss += loss.item()

            # compute backward pass
            loss.backward()

            # update model weights
            optimizer_closs.step()
            for param in self.centerloss_func.parameters():
                param.grad.data *= (1.0 / self.centerloss_weight)
            optimizer_closs.step()

            # delete mini-batch data from device
            del inputs
            del targets

        # calculate average loss across all mini-batches
        train_loss /= len(self.train_loader)

        return train_loss
