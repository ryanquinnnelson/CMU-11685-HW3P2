"""
Contains all Dataset objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

import glob
import os

from torch.utils.data import Dataset
from PIL import Image


class TestDataset(Dataset):
    """
    Defines image dataset for test data.
    """

    def __init__(self, test_dir, transform):
        # get all filenames in numerical order
        filenames = glob.glob(test_dir + '/*.jpg')
        filenames.sort(key=lambda e: int(os.path.basename(e).split('.')[0]))
        self.imgs = filenames
        self.length = len(filenames)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        f = self.imgs[index]

        # open image
        img = Image.open(f)  # .convert('RGB') #?? should I not be converting to RGB?

        # convert into a Tensor
        return self.transform(img)
