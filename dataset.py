from torch.utils.data import Dataset
import numpy as np
import torch

class XLCTCDataset(Dataset):
    def __init__(self, root_path, mode='train', **kwargs):
        self.mode = mode
        if self.mode == 'train':
            self.data = np.load(root_path + mode + '.npy', allow_pickle=True)
            # self.data2 = np.load(root_path + 'dev' + '.npy', allow_pickle=True)
            # self.data = np.concatenate((self.data, self.data2), 0)
            self.labels = np.load(root_path + mode + '_labels.npy', allow_pickle=True)
            # self.labels2 = np.load(root_path + 'dev' + '_labels.npy', allow_pickle=True)
            # self.labels = np.concatenate((self.labels, self.labels2), 0)
            self.data, self.labels = self.preprocessing(window_len=kwargs['window_length'])
        elif self.mode == 'dev':
            self.data = np.load(root_path + mode + '.npy', allow_pickle=True)
            self.labels = np.load(root_path + mode + '_labels.npy', allow_pickle=True)
            self.data, self.labels = self.preprocessing(window_len=kwargs['window_length'])
        else:
            self.data = np.load(root_path + mode + '.npy', allow_pickle=True)
            self.data = self.preprocessing(window_len=kwargs['window_length'])

    def preprocessing(self, window_len=150, scale=150):
        data = []
        label = []
        length = []
        label_length = []
        max = 0
        min = 100
        for utt in self.data:
            max = np.max(utt) if np.max(utt) > max else max
            min = np.min(utt) if np.min(utt) < min else min
        self.data = self.data / (max - min)
        for utt in self.data:
            step = utt.shape[0]
            utt = np.pad(utt, ((0, 3500-step),(0,0)), 'constant')
            data.append(utt)
            length.append(int(step))
        self.length = length
        self.data = data
        if self.mode == 'train' or self.mode == 'dev':
            for l in self.labels:
                label_step = len(l)
                l = np.pad(l, (0, 364 - label_step), 'constant')
                label.append(l)
                label_length.append(label_step)
            self.labels = label
            self.label_length = label_length
            return self.data, self.labels
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dic = dict()
        data_dic['data'] = torch.as_tensor(self.data[idx], dtype=torch.float32)
        data_dic['length'] = torch.as_tensor(self.length[idx], dtype=torch.int64)
        if self.mode == 'train' or self.mode == 'dev':
            data_dic['label'] = torch.as_tensor(self.labels[idx], dtype=torch.long)
            data_dic['label_length'] = torch.as_tensor(self.label_length[idx], dtype=torch.int64)
        return data_dic
