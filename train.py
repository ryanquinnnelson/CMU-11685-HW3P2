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
from Levenshtein import distance as lev
import os
from phoneme_list import PHONEME_LIST, PHONEME_MAP

PHONEME_MAP = np.array(PHONEME_MAP)
parser = argparse.ArgumentParser(description='Process some integers.')
# dataset args
parser.add_argument('--mode', default='train')
parser.add_argument('--window_length', default=30)
parser.add_argument('--root_path', default='./HW3P2_Data/')
parser.add_argument('--batch_size', default=64)
# model args
parser.add_argument('--nhead', default=1)
# training args
parser.add_argument('--max_epoch', default=50)
parser.add_argument('--lr', default=1e-3)
parser.add_argument('--weight_decay', default=5e-5)
parser.add_argument('--step', default=10)
parser.add_argument('--lr_decay', default=0.1)
parser.add_argument('--gpu', default=(0, 1))
parser.add_argument('--ckpt', default='./xxxx.pth')
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
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=kwargs['batch_size'], collate_fn=collate_fn)
    return dataloader


def collate_fn(batch_):
    max_length = 0
    max_label_length = 0
    length = []
    label_length = []
    data = []
    label = []
    for batch in batch_:
        max_length = batch['length'] if batch['length'] > max_length else max_length
        max_label_length = batch['label_length'] if batch['label_length'] > max_label_length else max_label_length
        length.append(batch['length'].unsqueeze(0))
        label_length.append(batch['label_length'].unsqueeze(0))
    for batch in batch_:
        data.append(batch['data'][:max_length,:].unsqueeze(0))
        label.append(batch['label'][:max_label_length].unsqueeze(0))
    batch = dict()
    batch['data'] = torch.cat(data, 0)
    batch['label'] = torch.cat(label, 0)
    batch['length'] = torch.cat(length, 0)
    batch['label_length'] = torch.cat(label_length, 0)
    return batch


if __name__ == '__main__':

    model = creat_model(**configs).cuda()
    if configs['mode'] == 'train_resume':
        model_path = configs['ckpt']
        state_dict = torch.load(model_path)['net']
        print(f'load model from {model_path}')
        model.load_state_dict(state_dict, strict=False)
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
    criterion = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs['step'], gamma=configs['lr_decay'])

    iter = 0
    best_acc = 0
    best_iter = 0
    best_loss = 100
    best_epoch = 0
    best_loss_epoch = 0
    best_dis = 100

    print('Begin Training')
    if configs['mode'] == 'train':
        max_iter = configs['max_epoch'] * len(dataset) / configs['batch_size']
        max_epoch = configs['max_epoch']
        with tqdm(total=int(max_iter)) as pbar:
            for epoch in range(max_epoch):
                scheduler.step()
                ############################################
                # Train
                ############################################
                for i, dic in enumerate(dataloader):
                    optimizer.zero_grad()
                    data = dic['data'].cuda()
                    length = dic['length']
                    label_length = dic['label_length']
                    label = dic['label']
                    label = label.cuda()
                    out = model(data, length)
                    loss = criterion(out, label, length, label_length)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    pbar.update(1)
                    iter += 1
                    pbar.set_description("loss: {}".format(loss.cpu().detach().item()))
                ############################################
                # Eval
                ############################################
                val_loss = 0
                val_dis = 0
                count = 0
                model.eval()
                for j, dic in enumerate(val_dataloader):
                    with torch.no_grad():
                        result = []
                        gt = []
                        data = dic['data']
                        count += data.shape[0]
                        data = data.cuda()
                        length = dic['length']
                        label_length = dic['label_length']
                        out = model(data, length)
                        label = dic['label']
                        label = label.cuda()
                        loss = criterion(out, label, length, label_length)
                        out = out.permute(1, 0, 2)

                        for b in range(configs['batch_size']):
                            try:
                                o = out[b].unsqueeze(0)[:, :length[b], :]

                                # TODO: use decoder here
                                raise NotImplementedError

                                result.append(''.join(PHONEME_MAP[res].tolist()))
                                g = label.cpu().numpy()[b][:label_length[b]]
                                gt.append(''.join(PHONEME_MAP[g].tolist()))
                                # print(len(res), len(g))
                            except:
                                # HACK! handle incomplete batch, you can use drop_last but I am lazy
                                pass

                        for a, b in zip(result, gt):
                            val_dis += lev(a, b)

                        val_loss += loss
                model.train()
                val_loss = val_loss / (count + 1e-5) * configs['batch_size']
                val_dis = val_dis / (count + 1e-5)
                if val_dis < best_dis:
                    best_dis = val_dis
                    best_epoch = epoch
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_loss_epoch = epoch
                print('\n')
                print('########################################################################')
                print("epoch: {0}, loss: {1}, dis: {2}, best_dis: {3}, best_epoch: {4}, best_loss_epoch: {5}"
                      .format(epoch, val_loss, val_dis, best_dis, best_epoch, best_loss_epoch))
                print('########################################################################')

                state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, './{0}__checkpoint.pth'.format(epoch))

    else:
        print('Use test.py to test')


