import torch
import torch.nn as nn

import pandas as pd

from rich import print
import dill as pickle


class Generator(nn.Module):
    """Generator class
    """
    def __init__(self, model=None):
        super().__init__()
        
        if model is None:
            # define neural network layers
            self.model = nn.Sequential(
                nn.Linear(1, 200),
                nn.Sigmoid(),
                nn.Linear(200, 784),
                nn.Sigmoid()
            )
        else:
            self.model = model
        
        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        # counter and accumulator for progress
        self.counter = 0;
        self.progress = []
    
    
    def forward(self, inputs):        
        # simply run model
        return self.model(inputs)
    
    
    def train(self, D, inputs, targets):
        # calculate the output of the network
        g_output = self.forward(inputs)
        
        # pass onto Discriminator
        d_output = D.forward(g_output)
        
        # calculate error
        loss = D.loss_function(d_output, targets)

        # increase counter and accumulate error every 10
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())

        # zero gradients, perform a backward pass, update weights
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
            'figsize':  (16, 8), 
            'ylim':  ylim,
            'xlim':  xlim, 
            'alpha':  0.1, 
            'marker':  '.', 
            'grid':  True, 
            'yticks': yticks
        }
        df.plot(**plt_kwargs)


    def pickle(self, filename='classifier.pkl'):
        """Pickle and save the trained generator.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)