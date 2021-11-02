"""
All things related to datasets that are not images.
"""
__author__ = 'ryanquinnnelson'

import logging

import numpy as np


# TODO: Remove slicing of dataset when ready
class NumericalDatasetHandler:
    def __init__(self, data_dir, train_data, train_labels, val_data, val_labels, test_data, train_class, val_class,
                 test_class, train_collate_fn, val_collate_fn, test_collate_fn):
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
        logging.info('Initializing numerical dataset handler...')

    def get_train_dataset(self):

        # load data
        data = np.load(self.train_data, allow_pickle=True)
        labels = np.load(self.train_labels, allow_pickle=True)
        logging.info(f'Loaded {len(data)} training records and {len(labels)} corresponding labels.')

        # initialize dataset
        dataset = self.train_class(data[:4], labels[:4])

        return dataset

    def get_val_dataset(self):

        # load data
        data = np.load(self.val_data, allow_pickle=True)
        labels = np.load(self.val_labels, allow_pickle=True)
        logging.info(f'Loaded {len(data)} validation records and {len(labels)} corresponding labels.')

        # initialize dataset
        dataset = self.val_class(data[:4], labels[:4])

        return dataset

    def get_test_dataset(self):

        # load data
        data = np.load(self.test_data, allow_pickle=True)
        logging.info(f'Loaded {len(data)} test records.')

        # initialize dataset
        dataset = self.test_class(data[:4])

        return dataset
