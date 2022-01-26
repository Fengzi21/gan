from pathlib import Path

import pandas  # pandas to read csv files
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset


datadir = Path("/public/liuyf/learnspace/makeyourownneuralnetwork/mnist_dataset/")


class MnistDataset(Dataset):
    """MNIST Dataset class.
    """
    def __init__(self, csv_file, using_cuda=False):
        """Read pandas DataFrame.
        """
        self.data_df = pandas.read_csv(csv_file, header=None)
        
        # use GPU if it is available
        if torch.cuda.is_available() and using_cuda:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            print("using cuda:", torch.cuda.get_device_name(0))
            self.device = torch.device("cuda")
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.device = torch.device("cpu")
            self.FloatTensor = torch.FloatTensor
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        # image target (label)
        label = self.data_df.iloc[index, 0]
        target = torch.zeros((10))
        target[label] = 1.0
        
        # image data, normalised from 0-255 to 0-1
        image_values = self.FloatTensor(self.data_df.iloc[index, 1:].values) / 255.0
        
        # return label, image data tensor and target tensor
        return label, image_values, target
    
    def plot_image(self, index):
        img = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title(f"label = {self.data_df.iloc[index, 0]}")
        plt.imshow(img, interpolation='none', cmap='Blues')