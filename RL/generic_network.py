from torch import nn, optim
from torch.distributions.normal import Normal
import torch
import os

from numpy import pi

from config import TrainingRun

PI = pi

# Custom weight initialization function
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.1)
        nn.init.normal_(m.bias, mean=0.0, std=0.1)

class GenericNetwork:
    
    def __init__(self, name="", save_path=None, config=None) -> None:

        if config is None:
            self.config = TrainingRun()

        # String name for identification
        self.name = name

        # Generic network properties... 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimiser = optim.Adam(self.parameters(), lr=self.config.q_learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimiser, step_size=100, gamma=0.99)
        self.to(self.device)

        self.default_folder = 'saved_networks'
        
        if save_path is None:
            self.save_path = f"{self.default_folder }/{name}_{self.__class__.__name__}_save.pt"
        else:
            self.save_path = save_path

        # TODO: Over complicated?
        # send all torch tensors to cuda
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(self.device))

    def save_network(self, folder=None, tag=""):
        """
        Save network
        """

        if folder is None:
            folder = self.default_folder

        if not os.path.exists(folder):
            os.makedirs(folder)

        save_path = f"{folder}/{tag}{self.name}_{self.__class__.__name__}_save.pt"
        with open(save_path, 'wb') as f:
            torch.save(self.state_dict(),f)

    def load_network(self,folder=None, tag=""):
        """
        Load a network
        """
        if folder is None:
            folder = self.default_folder

        save_path = f"{folder}/{tag}{self.name}_{self.__class__.__name__}_save.pt"
        self.load_state_dict(torch.load(save_path))