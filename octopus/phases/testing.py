"""
Defines the testing phase of model training.
"""
__author__ = 'ryanquinnnelson'

import logging
import torch
import numpy as np


class Testing:
    """
    Defines object to manage testing phase of training.
    """

    def __init__(self, test_loader, devicehandler):
        """
        Initialize Testing object.

        Args:
            test_loader (DataLoader): DataLoader for test data
            devicehandler (DeviceHandler): manages device on which training is being run
        """
        logging.info('Loading testing phase...')
        self.test_loader = test_loader
        self.devicehandler = devicehandler

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

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for i, batch in enumerate(self.test_loader):

                if type(batch) is tuple:
                    # loader contains inputs and targets
                    inputs = batch[0]
                    targets = batch[1]
                else:
                    # loader contains only inputs
                    inputs = batch
                    targets = None

                # prep
                inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

                # forward pass
                out = model.forward(inputs)

                # capture output for mini-batch
                out = out.cpu().detach().numpy()  # extract from gpu if necessary
                output.append(out)

        combined = np.concatenate(output, axis=0)

        return combined
