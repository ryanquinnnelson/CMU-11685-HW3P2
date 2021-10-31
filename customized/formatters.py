"""
Contains all Formatter objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

import pandas as pd
import numpy as np
import json
import logging
import os


def _convert_output(out):
    """
    Convert 2D output to 1D a single class label.

    Args:
        out (np.array): 2D output in which each row is a datapoint and each column is a single class

    Returns: np.array 1D output

    """
    out = np.argmax(out, axis=1)  # column with max value in each row is the index of the predicted label

    return out


class OutputFormatter:
    """
    Defines an object to manage formatting of test output.
    """
    def __init__(self, data_dir):
        """
        Initialize OutputFormatter.
        Args:
            data_dir (str): fully-qualified path to data directory
        """
        self.data_dir = data_dir

    def _read_class_to_idx_json(self):
        """
        Read the class_to_idx mapping from ImageFolder saved in the data directory.

        Returns: json mapping

        """
        source = os.path.join(self.data_dir, 'class_to_idx.json')
        logging.info(f'Reading class_to_idx mapping from {source}...')
        with open(source) as mapping_source:
            mapping = json.load(mapping_source)
            return mapping

    def format_output(self, out):
        """
        Format given model output as desired.

        Args:
            out (np.array): Model output

        Returns: DataFrame after formatting

        """
        labels = _convert_output(out)

        # add an index column
        df = pd.DataFrame(labels).reset_index(drop=False)

        # change column names
        df = df.rename(columns={0: "output_label", 'index': 'idprefix'})

        # add .jpg to the id column
        df = df.astype({'idprefix': 'str'})
        df['idsuffix'] = '.jpg'
        df['id'] = df['idprefix'] + df['idsuffix']

        # drop extra columns generated
        df = df.drop(['idprefix', 'idsuffix'], axis=1)

        # remap output labels to correct labels based on ImageFolder (see Note 1 in imagedatasethandler)
        mapping = self._read_class_to_idx_json()
        df['label'] = df.apply(lambda x: list(mapping.keys())[list(mapping.values()).index(x['output_label'])], axis=1)

        # ensure id is first column
        df = df[['id', 'label', 'output_label']]

        return df
