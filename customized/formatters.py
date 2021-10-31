"""
Contains all Formatter objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

import pandas as pd
import numpy as np
import json
import logging
import os
from customized.helper import convert_to_phonemes, target_to_phonemes, convert_to_string, decode_output


class OutputFormatter:
    """
    Defines an object to manage formatting of test output.
    """

    def __init__(self, data_dir, ctcdecodehandler):
        """
        Initialize OutputFormatter.
        Args:
            data_dir (str): fully-qualified path to data directory
        """
        self.data_dir = data_dir
        self.ctcdecode = ctcdecodehandler.get_ctcdecoder()

    def format_output(self, out):
        """
        Format given model output as desired.

        Args:
            out (np.array): Model output

        Returns: DataFrame after formatting

        """
        # labels = _convert_output(out)
        #
        # # add an index column
        # df = pd.DataFrame(labels).reset_index(drop=False)
        #
        # # change column names
        # df = df.rename(columns={0: "output_label", 'index': 'idprefix'})
        #
        # # add .jpg to the id column
        # df = df.astype({'idprefix': 'str'})
        # df['idsuffix'] = '.jpg'
        # df['id'] = df['idprefix'] + df['idsuffix']
        #
        # # drop extra columns generated
        # df = df.drop(['idprefix', 'idsuffix'], axis=1)
        #
        # # remap output labels to correct labels based on ImageFolder (see Note 1 in imagedatasethandler)
        # mapping = self._read_class_to_idx_json()
        # df['label'] = df.apply(lambda x: list(mapping.keys())[list(mapping.values()).index(x['output_label'])], axis=1)
        #
        # # ensure id is first column
        # df = df[['id', 'label', 'output_label']]

        return df
