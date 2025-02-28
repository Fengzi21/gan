import numpy as np
import matplotlib.pyplot as plt
import h5py

import torch
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    """CelebA Dataset class.
    """
    def __init__(self, file):
        self.file_object = h5py.File(file, 'r')
        self.dataset = self.file_object['img_align_celeba']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index >= len(self.dataset):
            raise IndexError()
        img = np.array(self.dataset[f'{index}.jpg'])
        return torch.cuda.FloatTensor(img) / 255.0

    def plot_image(self, index):
        plt.imshow(np.array(self.dataset[f'{index}.jpg']), interpolation='nearest')
