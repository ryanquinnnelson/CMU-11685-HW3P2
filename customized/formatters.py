"""
Contains all Formatter objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

import pandas as pd
import logging
from customized.helper import out_to_phonemes, target_to_phonemes, convert_to_string, decode_output
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

        logging.info(f'out shape:{out.shape}')

        # decode output
        # out: (N_TIMESTEPS x BATCHSIZE x N_LABELS)
        beam_results, beam_scores, timesteps, out_lens = decode_output(out, ctcdecode)

        # convert to strings using phoneme map (not phoneme list)
        n_batches = beam_results.shape[0]
        logging.info(f'Converting {n_batches} beam results to phonemes...')

        for i in range(n_batches):
            out_converted = out_to_phonemes(i, beam_results, out_lens, pl.PHONEME_MAP)
            logging.info(f'out_converted[{i}]:{out_converted}')

            converted_str = convert_to_string(out_converted)
            logging.info(f'converted_str[{i}]:{converted_str}')
            converted.append(converted_str)

        # convert string array to dataframe
        df = pd.DataFrame(converted).reset_index(drop=False)
        logging.info('dataframe')
        logging.info(f'\n{df.head()}')
        logging.info(df.columns)

        # change column names
        df = df.rename(columns={0: 'label', 'index': 'id'})
        logging.info(f'\n{df.head()}')

        return df
