from torch import nn
import torch
import numpy as np
from dataset import XLCTCDataset
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from model import XLCTCmodel
import pandas as pd
from ctcdecode import CTCBeamDecoder
import os
from phoneme_list import PHONEME_MAP

PHONEME_MAP = np.array(PHONEME_MAP)
parser = argparse.ArgumentParser(description='Process some integers.')
# dataset args
parser.add_argument('--mode', default='test')
parser.add_argument('--window_length', default=30)
parser.add_argument('--root_path', default='./HW3P2_Data/')
parser.add_argument('--batch_size', default=1)
args = parser.parse_args()
configs = vars(args)
print(configs)


def creat_model(**kwargs):
    model = XLCTCmodel(**kwargs)
    return model


def create_dataset(val=False, **kwargs):
    if val:
        return XLCTCDataset(root_path=kwargs['root_path'], mode='dev',
                         window_length=kwargs['window_length'])
    return XLCTCDataset(root_path=kwargs['root_path'], mode=kwargs['mode'],
                     window_length=kwargs['window_length'])


def create_dataloader(dataset, **kwargs):
    shuffle = True if kwargs['mode'] == 'train' else False
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=kwargs['batch_size'])
    return dataloader


if __name__ == '__main__':
    model = creat_model(**configs)
    assert configs['mode'] == 'test', 'test.py only support test mode'
    if configs['mode'] == 'test':
        model_path = './6__checkpoint.pth'
        state_dict = torch.load(model_path)['net']
        print(f'load model from {model_path}')
        model.load_state_dict(state_dict)
    dataset = create_dataset(**configs)
    dataloader = create_dataloader(dataset, **configs)
    loader = iter(dataloader)
    val_dataset = create_dataset(val=True, **configs)
    val_dataloader = create_dataloader(val_dataset, **configs)
    val_loader = iter(val_dataloader)
    # Default setting copied from "https://github.com/parlance/ctcdecode"
    decoder = CTCBeamDecoder(
        PHONEME_MAP,
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=0,
        log_probs_input=True
    )

    print('Begin Testing')

    count = 0
    result = []
    model.eval()

    with tqdm(total=len(dataset) / configs['batch_size']) as pbar:
        for j, val_dic in enumerate(dataloader):
            with torch.no_grad():
                count += 1
                data = val_dic['data']
                data = data
                length = val_dic['length']
                out = model(data, length)
                out = out.permute(1, 0, 2)

                assert configs['batch_size'] == 1, 'use bs 1'
                for b in range(configs['batch_size']):

                    # TODO: use decoder here
                    raise NotImplementedError

                    result.append(''.join(PHONEME_MAP[res].tolist()))
                pbar.update(1)
        assert len(result) == len(dataset), f'result {len(result)} should have the same ' \
                                            f'length as input {len(dataset)}'
        df = pd.DataFrame({'id': [i for i in range(len(result))], 'label': result}, columns=['id', 'label'])
        df.to_csv("result.csv", index=False)
