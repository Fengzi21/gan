import torch
import torch.nn as nn

import pandas as pd

from rich import print
import dill as pickle  # for pickling the trained neural network


class Classifier(nn.Module):
    """PyTorch MNIST dataset classifier class.
    """
    def __init__(self, model=None):
        # initialise parent pytorch class
        super().__init__()

        if model is None:
            # define neural network layers
            self.model = nn.Sequential(nn.Linear(784, 200), nn.Sigmoid(), nn.Linear(200, 10), nn.Sigmoid())
        else:
            self.model = model

        # create loss function
        self.loss_function = nn.MSELoss()

        # create optimiser, using simple stochastic gradient descent
        # pass the learnable parameters and set learning rate to 0.01
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        # counter and accumulator for progress
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        """Simply run model to feed signal forward and get outputs.
        """
        return self.model(inputs)

    def train(self, inputs, targets, print_counter=False):
        """Train the classifier.
        """
        outputs = self.forward(inputs)  # calculate the output of the network
        loss = self.loss_function(outputs, targets)  # calculate loss

        self.counter += 1  # increase counter

        # accumulate error every 10
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        # print counter every 10000 loops
        if print_counter and (self.counter % 10000 == 0):
            print('counter = ', self.counter)

        # zero gradients, perform a backward pass, and update the weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self, xlim=(None, None), ylim=(None, None), yticks=None):
        """Plot classifier error.
        """
        df = pd.DataFrame(self.progress, columns=['loss'])
        if ylim == (None, None):
            ylim = (min(self.progress), max(self.progress))
        plt_kwargs = {
            'figsize': (16, 8),
            'ylim': ylim,
            'xlim': xlim,
            'alpha': 0.1,
            'marker': '.',
            'grid': True,
            'yticks': yticks,
        }
        df.plot(**plt_kwargs)

    def pickle(self, filename='classifier.pkl'):
        """Pickle and save the trained classifier.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
