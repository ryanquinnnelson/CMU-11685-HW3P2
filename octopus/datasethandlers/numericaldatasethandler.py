"""
All things related to datasets that require customized classes for Training, Validation, and Testing data.
"""
__author__ = 'ryanquinnnelson'

import logging

import numpy as np


class NumericalDatasetHandler:
    def __init__(self, data_dir, train_data, train_labels, val_data, val_labels, test_data, train_class, val_class,
                 test_class, train_collate_fn, val_collate_fn, test_collate_fn):
        """
        Initialize NumericalDatasetHandler.

        :param data_dir (str): Fully-qualified path that is the root of data subdirectories.
        :param train_data (str): Fully-qualified path to training data.
        :param train_labels (str): Fully-qualified path to training labels.
        :param val_data (str):  Fully-qualified path to validation data.
        :param val_labels (str):  Fully-qualified path to validation labels.
        :param test_data (str):  Fully-qualified path to test data.
        :param train_class (Dataset): Dataset class representing training data.
        :param val_class (Dataset):  Dataset class representing validation data.
        :param test_class (Dataset):  Dataset class representing test data.
        :param train_collate_fn (function): Collate function for training data.
        :param val_collate_fn (function):  Collate function for validation data.
        :param test_collate_fn (function):  Collate function for test data.
        """
        logging.info('Initializing numerical dataset handler...')

        self.data_dir = data_dir
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.test_data = test_data
        self.train_class = train_class
        self.val_class = val_class
        self.test_class = test_class
        self.train_collate_fn = train_collate_fn
        self.val_collate_fn = val_collate_fn
        self.test_collate_fn = test_collate_fn

    def get_train_dataset(self):
        """
        Load training data into memory and initialize the Dataset object.
        :return: Dataset
        """

        # load data
        data = np.load(self.train_data, allow_pickle=True)
        labels = np.load(self.train_labels, allow_pickle=True)
        logging.info(f'Loaded {len(data)} training records and {len(labels)} corresponding labels.')

        # initialize dataset
        dataset = self.train_class(data, labels)

        return dataset

    def get_val_dataset(self):
        """
        Load validation data into memory and initialize the Dataset object.
        :return: Dataset
        """

        # load data
        data = np.load(self.val_data, allow_pickle=True)
        labels = np.load(self.val_labels, allow_pickle=True)
        logging.info(f'Loaded {len(data)} validation records and {len(labels)} corresponding labels.')

        # initialize dataset
        dataset = self.val_class(data, labels)

        return dataset

    def get_test_dataset(self):
        """
        Load test data into memory and initialize the Dataset object.
        :return: Dataset
        """

        # load data
        data = np.load(self.test_data, allow_pickle=True)
        logging.info(f'Loaded {len(data)} test records.')

        # initialize dataset
        dataset = self.test_class(data)

        return dataset
