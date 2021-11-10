"""
Defines helper functions for customized module.
"""
__author__ = 'ryanquinnnelson'

import logging

import numpy as np
import torch
from pynvml import *

from customized import phoneme_details as pl


def check_status():
    """
    Check the GPU memory.

    https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    :return: None
    """
    # check gpu properties
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    logging.info(f'total_memory:{t}')
    logging.info(f'free inside reserved:{f}')

    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    logging.info(f'total    : {info.total}')
    logging.info(f'free     : {info.free}')
    logging.info(f'used     : {info.used}')


def out_to_phonemes(k, beam_results, out_lens, p_map):
    """
    Convert beam results of kth output to a List of phoneme encodings.

    :param k (int): Index of output to decode.
    :param beam_results (Tensor): CTCBeamDecoder beam results. Has shape (BATCHSIZE x N_BEAMS X N_TIMESTEPS).
    :param out_lens (Tensor): output lengths for CTCBeamDecoder beam results. Has shape (BATCHSIZE x N_BEAMS).
    :param p_map (List): List mapping beam integer values to phoneme encodings, indexed from 0.
    :return: List(str) of phoneme encodings
    """
    # k is the kth item in the batch, indexed from 0
    j = 0  # top jth beam, where 0 is the highest
    beam_result = beam_results[k][0][:out_lens[k][0]]  # (BEAM_LEN,)
    beam_result = beam_result.numpy()

    # convert beam indexes to phonemes
    converted = np.array([p_map[idx] for idx in beam_result.flatten()])  # (BEAM_LEN,)

    return converted


def target_to_phonemes(target, p_map):
    """
    Convert target indexes to a List of phoneme encodings.

    :param target (Tensor): Target phoneme indexes for a single record. Has shape (UTTERANCE_LABEL_LENGTH,)
    :param p_map (List): List mapping beam integer values to phoneme encodings, indexed from 0.
    :return: List(str) of phoneme encodings
    """
    target = target.numpy()

    # convert indexes to phonemes
    converted = np.array([p_map[idx] for idx in target.flatten()])

    return converted


def convert_to_string(converted_list):
    """
    Concatenate List of strings into a single string.

    :param converted_list (List): List of strings
    :return: String representing phoneme encodings
    """
    return ''.join(converted_list.tolist()).strip()


def decode_output(out, ctcdecoder, out_lengths):
    """
    Converts model output into strings representing phoneme encodings.

    :param out (Tensor): model output. Represents log probabilities of each of the 42 phoneme labels at each timestep. Shape is (N_TIMESTEPS,BATCHSIZE,N_LABELS).
    :param ctcdecoder (CTCBeamDecoder): decoder
    :param out_lengths:
    :param i (int): Batch index, indexed from 0.
    :return: Tuple(Tensor,Tensor,Tensor,Tensor) representing (beam_results, beam_scores, timesteps, out_lens)
    """
    # transpose output to fit expectations of CTCBeamDecoder
    out = torch.transpose(out, 0, 1)  # (BATCHSIZE,N_TIMESTEPS,N_LABELS)

    # for each record, take log probabilities of each of the 42 phoneme labels at each timestep
    # extract beams from this output representing the most probable list of phonemes for utterance
    beam_results, beam_scores, timesteps, out_lens = ctcdecoder.decode(out, seq_lens=out_lengths)

    return beam_results, beam_scores, timesteps, out_lens


def calculate_distances(beam_results, out_lens, targets):
    """
    Given the decoded output, calculate the Levenshtein distance between predicted and target phoneme strings for each
    record in the batch.

    :param beam_results (Tensor): CTCBeamDecoder beam results. Has shape (BATCHSIZE x N_BEAMS X N_TIMESTEPS).
    :param out_lens (Tensor): output lengths for CTCBeamDecoder beam results. Has shape (BATCHSIZE x N_BEAMS).
    :param targets (Tensor): Target phoneme indexes for each record in the batch. Has shape (BATCHSIZE,UTTERANCE_LABEL_LENGTH)
    :param i (int): Batch index, indexed from 0.
    :return: int representing the sum of distances of all records in the batch
    """
    import Levenshtein
    n_batches = beam_results.shape[0]
    total_distance = 0

    # calculate distance for each record in the batch
    for k in range(n_batches):
        # convert to phoneme list
        out_converted = out_to_phonemes(k, beam_results, out_lens, pl.PHONEME_MAP)  # (BEAM_LEN,)
        target_converted = target_to_phonemes(targets[k], pl.PHONEME_MAP)  # (TARGET_LEN,)

        # convert phoneme lists to single strings
        out_str = convert_to_string(out_converted)
        target_str = convert_to_string(target_converted)

        # calculate distance
        distance = Levenshtein.distance(out_str.strip(), target_str.strip())
        total_distance += distance

    return total_distance
