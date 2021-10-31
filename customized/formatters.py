"""
Contains all Formatter objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

import pandas as pd
import logging
from customized.helper import convert_to_phonemes, target_to_phonemes, convert_to_string, decode_output
import customized.phoneme_list as pl


class OutputFormatter:
    """
    Defines an object to manage formatting of test output.
    """

    def __init__(self):
        """
        Initialize OutputFormatter.
        Args:
            data_dir (str): fully-qualified path to data directory
        """
        logging.info('Initializing output formatter...')

    def format_output(self, out, ctcdecode):
        """
        Format given model output as desired.

        Args:
            out (np.array): Model output

        Returns: DataFrame after formatting

        """

        converted = []

        logging.info(f'out shape:{ out.shape}')

        # decode output
        beam_results, beam_scores, timesteps, out_lens = decode_output(out, ctcdecode)

        # convert to strings using phoneme map
        n_batches = beam_results.shape[0]
        logging.info(f'Converting {n_batches} beam results to phonemes...')

        for i in range(n_batches):
            out_converted = convert_to_phonemes(i, beam_results, out_lens, pl.PHONEME_MAP)
            logging.info(f'out_converted:{out_converted}')
            converted.append(out_converted)

        # convert string array to dataframe
        df = pd.DataFrame(converted).reset_index(drop=False)
        logging.info(df.head())
        logging.info(df.columns)

        # change column names
        df = df.rename(columns={0: "label", 'index': 'id'})
        logging.info(df.head())

        return df
