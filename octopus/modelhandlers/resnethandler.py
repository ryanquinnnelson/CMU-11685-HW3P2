"""
All things related to handling resnets.
"""
__author__ = 'ryanquinnnelson'

import logging

import octopus.models.resnets as resnets


class ResnetHandler:
    """
    Object to handle constructing resnets.
    """
    def __init__(self, model_type, in_features, num_classes):
        """
        Initialize ResnetHandler.

        Args:
            model_type (str): model to construct
            in_features (int): Number of channels in the input data
            num_classes (int): Number of classes in the classification
        """
        logging.info('Initializing model handling...')
        self.model_type = model_type
        self.in_features = in_features
        self.num_classes = num_classes

    def get_model(self):
        """
        Initialize the model according to the parameters.

        Returns: nn.Module model

        """
        logging.info('Initializing model...')
        model = None

        if self.model_type == 'Resnet18':
            model = resnets.Resnet18(self.in_features, self.num_classes)

        elif self.model_type == 'Resnet34':
            model = resnets.Resnet34(self.in_features, self.num_classes)

        elif self.model_type == 'Resnet34_v2':
            model = resnets.Resnet34_v2(self.in_features, self.num_classes)

        elif self.model_type == 'Resnet34_v3':
            model = resnets.Resnet34_v3(self.in_features, self.num_classes)

        elif self.model_type == 'Resnet34_v4':
            model = resnets.Resnet34_v4(self.in_features, self.num_classes)

        elif self.model_type == 'Resnet34_v5':
            model = resnets.Resnet34_v5(self.in_features, self.num_classes)

        elif self.model_type == 'Resnet50':
            model = resnets.Resnet50(self.in_features, self.num_classes)

        elif self.model_type == 'Resnet101':
            model = resnets.Resnet101(self.in_features, self.num_classes)

        elif self.model_type == 'Resnet152':
            model = resnets.Resnet152(self.in_features, self.num_classes)

        logging.info(f'Model initialized:\n{model}')
        return model


class ResnetHandlerCenterLoss:
    """
    Defines an object to handle constructing resnets that will be measured using centerloss.
    """
    def __init__(self, model_type, in_features, num_classes, feat_dim):
        """
        Initialize ResnetHandlerCenterLoss.

        Args:
            model_type (str): model to construct
            in_features (int): Number of channels in the input data
            num_classes (int): Number of classes in the classification
            feat_dim (int): number of features in the embedding
        """
        logging.info('Initializing model handling...')
        self.model_type = model_type
        self.in_features = in_features
        self.num_classes = num_classes
        self.feat_dim = feat_dim

    def get_model(self):
        """
        Initialize the model according to the parameters.

        Returns: nn.Module model

        """
        logging.info('Initializing model...')
        model = None

        if self.model_type == 'Resnet18':
            model = resnets.Resnet18(self.in_features, self.num_classes, self.feat_dim)

        elif self.model_type == 'Resnet34':
            model = resnets.Resnet34(self.in_features, self.num_classes, self.feat_dim)

        logging.info(f'Model initialized:\n{model}')
        return model
