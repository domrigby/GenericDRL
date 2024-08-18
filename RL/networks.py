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
            config = TrainingRun()

        # String name for identification
        self.name = name

        # Generic network properties... 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimiser = optim.Adam(self.parameters(), lr=config.LearningRate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimiser, step_size=10, gamma=0.99)
        self.to(self.device)
        
        if save_path is None:
            self.save_path = f"saved_networks/{name}_{self.__class__.__name__}_save.pt"
        else:
            self.save_path = save_path

        # TODO: Over complicated?
        # send all torch tensors to cuda
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(self.device))

    def save_network(self, folder='saved_networks', tag=""):
        """ Save network
        """

        if not os.path.exists(folder):
            os.makedirs(folder)

        save_path = f"{folder}/{tag}{self.name}_{self.__class__.__name__}_save.pt"
        with open(save_path, 'wb') as f:
            torch.save(self.state_dict(),f)

    def load_network(self,folder='saved_networks', tag=""):
        """Load a network
        """
        save_path = f"{folder}/{tag}{self.name}_{self.__class__.__name__}_save.pt"
        self.load_state_dict(torch.load(save_path))

class ActorNetwork(nn.Module, GenericNetwork):

    def __init__(self, state_dim, action_dim, name="", max_action=1) -> None:

        # No Super so can initialise at different times
        nn.Module.__init__(self)

        hidden_lay_1 = 256
        hidden_lay_2 = 256

        self.kin_fc1 = nn.Linear(state_dim, hidden_lay_1)
        self.kin_fc2 = nn.Linear(hidden_lay_1, hidden_lay_2)

        self.mean = nn.Linear(hidden_lay_2, action_dim)
        self.sigma = nn.Linear(hidden_lay_2, action_dim)

        self.noise = 1e-6

        self.act = nn.LeakyReLU()

        self.max_action = max_action

        GenericNetwork.__init__(self, name, None)

        self.apply(init_weights)

    def forward(self, state):

        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        x = self.act(self.kin_fc1(state))
        x = self.act(self.kin_fc2(x))

        # Get mean and std 
        mean = self.mean(x)
        std = self.sigma(x)

        std = torch.clamp(std, min=self.noise, max=1)

        return mean, std
    
    def sample_actions(self, state, reparameterize=False):

        # run network
        mean, std = self.forward(state)
        action_distribution = Normal(mean, std)

        # sample the dsitribtion
        if reparameterize:
            actions = action_distribution.rsample()
        else:
            actions = action_distribution.sample()

        # Convert infinite range Gaussian into finite tanh distribution
        action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)

        # Take the log of the actions for entropy
        log_probs = action_distribution.log_prob(actions)
        log_probs -= torch.log(torch.tensor(self.max_action).to(self.device)) + torch.log((1-torch.tanh(actions).pow(2))+self.noise)

        if len(log_probs.size()) > 1:
            log_probs = log_probs.sum(1, keepdim=True)
        else:
            log_probs = log_probs.sum()

        return action, log_probs

class ValueNetwork(nn.Module, GenericNetwork):

    def __init__(self, state_dim, name="") -> None:
        nn.Module.__init__(self)

        hidden_lay_1 = 256
        hidden_lay_2 = 256

        self.fc1 = nn.Linear(state_dim, hidden_lay_1)
        self.fc2 = nn.Linear(hidden_lay_1, hidden_lay_2)

        self.value = nn.Linear(hidden_lay_2, 1)

        self.act = nn.ReLU()

        GenericNetwork.__init__(self, name, None)

    def forward(self, state):
        x = self.act(self.fc1(state))
        x = self.act(self.fc2(x))
        value = self.value(x)
        return value
    

class CriticNetwork(nn.Module, GenericNetwork):

    def __init__(self, state_dim, action_dim, name="") -> None:
        nn.Module.__init__(self)

        hidden_lay_1 = 256
        hidden_lay_2 = 256

        self.fc1 = nn.Linear(state_dim+action_dim, hidden_lay_1)
        self.fc2 = nn.Linear(hidden_lay_1, hidden_lay_2)

        self.q_val = nn.Linear(hidden_lay_2, 1)

        self.act = nn.ReLU()

        GenericNetwork.__init__(self, name, None)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        q = self.q_val(x)
        return q