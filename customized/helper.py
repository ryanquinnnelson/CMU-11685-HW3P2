import logging

import numpy as np
import torch

from customized import phoneme_details as pl
from pynvml import *


def check_status():
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


def out_to_phonemes(i, beam_results, out_lens, p_map):
    """

    :param i:
    :param beam_results:
    :param out_lens:
    :param p_map: phoneme map
    :return:
    """
    # i is the ith item in the batch, indexed from 0
    j = 0  # top jth beam, where 0 is the highest
    beam = beam_results[i][j][:out_lens[i][j]]  # (BEAM_LEN,)
    beam = beam.numpy()

    # convert beam indexes to phonemes
    converted = np.array([p_map[idx] for idx in beam.flatten()])  # (BEAM_LEN,)
    return converted


def target_to_phonemes(target, p_map):
    target = target.numpy()
    converted = np.array([p_map[idx] for idx in target.flatten()])
    return converted


def convert_to_string(converted_list):
    return ''.join(converted_list.tolist())


def decode_output(out, ctcdecode):
    out = torch.transpose(out, 0, 1)  # BATCHSIZE x N_TIMESTEPS x N_LABELS

    # decode batch
    beam_results, beam_scores, timesteps, out_lens = ctcdecode.decode(out)
    # logging.info(f'out transposed:{out.shape}')
    # logging.info(f'beam_results:{beam_results.shape}')  # BATCHSIZE x N_BEAMS X N_TIMESTEPS
    # logging.info(f'beam_scores:{beam_scores.shape}')  # BATCHSIZE x N_BEAMS
    # logging.info(f'timesteps:{timesteps.shape}')  # BATCHSIZE x N_BEAMS
    # logging.info(f'out_lens:{out_lens.shape},{out_lens}')  # BATCHSIZE x N_BEAMS

    return beam_results, beam_scores, timesteps, out_lens


def calculate_distances(beam_results, out_lens, targets):
    import Levenshtein
    n_batches = beam_results.shape[0]
    # logging.info(f'Calculating distance for {n_batches} entries...')
    total_distance = 0

    # calculate distance for each record in the batch
    for i in range(n_batches):
        out_converted = out_to_phonemes(i, beam_results, out_lens, pl.PHONEME_MAP)  # (BEAM_LEN,)
        target_converted = target_to_phonemes(targets[i], pl.PHONEME_MAP)  # (TARGET_LEN,)

        out_str = convert_to_string(out_converted)
        target_str = convert_to_string(target_converted)

        distance = Levenshtein.distance(out_str, target_str)
        total_distance += distance

        # logging.info(f'out_converted:{out_converted}')
        # logging.info(f'out_str:{out_str}')
        # logging.info(f'targets[{i}]:{targets[i]}')
        # logging.info(f'target_converted:{target_converted}')
        # logging.info(f'target_str:{target_str}')
        # logging.info(f'distance:{distance}')

    return total_distance
