"""
Everything related to using CTCBeamDecoder.
https://github.com/parlance/ctcdecode
"""
__author__ = 'ryanquinnnelson'

import logging
import multiprocessing
import customized.phoneme_details as pl


def get_ctcdecoder(labels, model_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width, num_processes, blank_id,
                   log_probs_input):
    """
    Initialize CTCBeamDecoder object with given parameters.

    :param labels (List): List of the tokens you used to train your model. They should be in the same order as your outputs. For example if your tokens are the english letters and you used 0 as your blank token, then you would pass in List("_abcdefghijklmopqrstuvwxyz").
    :param model_path (str): path to your external kenlm language model(LM). Default is None.
    :param alpha (float): Weighting associated with the LMs probabilities. A weight of 0 means the LM has no effect.
    :param beta (float): Weight associated with the number of words within our beam.
    :param cutoff_top_n (int): Cutoff number in pruning. Only the top cutoff_top_n characters with the highest probability in the vocab will be used in beam search.
    :param cutoff_prob (float): Cutoff probability in pruning. 1.0 means no pruning.
    :param beam_width (int): This controls how broad the beam search is.
    :param num_processes (int): Parallelize the batch using num_processes workers.
    :param blank_id (int): This should be the index of the CTC blank token (probably 0).
    :param log_probs_input (Boolean): If your outputs have passed through a softmax and represent probabilities, this should be false, if they passed through a LogSoftmax and represent negative log likelihood, you need to pass True.
    :return: CTCBeamDecoder
    """
    import ctcdecode

    # vals = [labels, model_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width, num_processes, blank_id,
    #                log_probs_input]
    # logging.info(f'decoder values:{vals}')
    d = ctcdecode.CTCBeamDecoder(labels=labels, model_path=model_path, alpha=alpha, beta=beta,
                                 cutoff_top_n=cutoff_top_n, cutoff_prob=cutoff_prob, beam_width=beam_width,
                                 num_processes=num_processes,
                                 blank_id=blank_id, log_probs_input=log_probs_input)
    return d


class CTCDecodeHandler:
    """
    Handles initializing CTCBeamDecoder objects.
    """

    def __init__(self, model_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width, blank_id,
                 log_probs_input):
        """
        Initialize CTCDecodeHandler

        :param model_path (str): path to your external kenlm language model(LM). Default is None.
        :param alpha (float): Weighting associated with the LMs probabilities. A weight of 0 means the LM has no effect.
        :param beta (float): Weight associated with the number of words within our beam.
        :param cutoff_top_n (int): Cutoff number in pruning. Only the top cutoff_top_n characters with the highest probability in the vocab will be used in beam search.
        :param cutoff_prob (float): Cutoff probability in pruning. 1.0 means no pruning.
        :param beam_width (int): This controls how broad the beam search is.
        :param blank_id (int): This should be the index of the CTC blank token (probably 0).
        :param log_probs_input (Boolean): If your outputs have passed through a softmax and represent probabilities, this should be false, if they passed through a LogSoftmax and represent negative log likelihood, you need to pass True.
        """
        logging.info('Initializing ctcdecode handler...')

        self.model_path = None if model_path == 'None' else model_path
        self.alpha = alpha
        self.beta = beta
        self.cutoff_top_n = cutoff_top_n
        self.cutoff_prob = cutoff_prob
        self.beam_width = beam_width  # beam_width=1 (greedy search); beam_width>1 (beam search)
        self.blank_id = blank_id
        self.log_probs_input = log_probs_input

        # calculating additional parameters needed for initialization
        n_cpus = multiprocessing.cpu_count()
        logging.info(f'CPU count:{n_cpus}.')
        self.num_processes = n_cpus
        self.labels = pl.PHONEME_MAP

    def ctcdecoder(self):
        """
        Initialize CTCBeamDecoder.
        :return: CTCBeamDecoder
        """
        return get_ctcdecoder(self.labels, self.model_path, self.alpha, self.beta, self.cutoff_top_n,
                              self.cutoff_prob, self.beam_width, self.num_processes, self.blank_id,
                              self.log_probs_input)
