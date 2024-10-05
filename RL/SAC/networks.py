from torch import nn, optim
from torch.distributions.normal import Normal
import torch
import os

from RL.generic_network import GenericNetwork

from numpy import pi

from config import TrainingRun

PI = pi

class ActorNetwork(nn.Module, GenericNetwork):

    def __init__(self, state_dim, action_dim, name="", min_action=1, max_action=1) -> None:

        # No Super so can initialise at different times
        nn.Module.__init__(self)

        hidden_lay_1 = 256
        hidden_lay_2 = 256

        self.kin_fc1 = nn.Linear(state_dim, hidden_lay_1)
        self.kin_fc2 = nn.Linear(hidden_lay_1, hidden_lay_2)

        self.mean = nn.Linear(hidden_lay_2, action_dim)
        self.log_sigma = nn.Linear(hidden_lay_2, action_dim)

        self.noise = 1e-6

        self.act = nn.LeakyReLU()

        self.std= 0

        GenericNetwork.__init__(self, name, None)

        self.action_bias = torch.tensor((max_action+min_action)/2.0, device=self.device)
        self.action_range = torch.tensor((max_action-min_action)/2.0, device=self.device)

        if self.config.policy_learning_rate != self.config.q_learning_rate:
            self.optimiser = optim.Adam(self.parameters(), lr=self.config.q_learning_rate)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimiser, step_size=100, gamma=0.99)

    def forward(self, state):

        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        x = self.act(self.kin_fc1(state))
        x = self.act(self.kin_fc2(x))

        # Get mean and std 
        mean = self.mean(x)
        log_std = self.log_sigma(x)

        log_std = torch.tanh(log_std)
        log_std = -5 + 0.5 * 7 * (log_std + 1)

        std = log_std.exp()

        self.std = std

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
        action = (self.action_range*torch.tanh(actions) + self.action_bias).to(self.device) 

        # Take the log of the actions for entropy
        log_probs = action_distribution.log_prob(actions)
        log_probs -= torch.log(self.action_range*(1-torch.tanh(actions).pow(2))+self.noise)

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