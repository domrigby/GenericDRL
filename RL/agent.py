from RL.memory import ReplayBuffer
import torch
from torch import nn, optim

from itertools import count
from datetime import datetime

from abc import abstractmethod
from typing import List

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from RL.generic_network import GenericNetwork

import numpy as np

from config import TrainingRun

class Agent:
    def __init__(self, network):
        # Place holder for future potential multi-agent
        self.actor_network = network

class RLAgent:

    def __init__(self, env, learning: bool = True, load_networks_dir: str = None, 
                 max_steps: int =1000, config: TrainingRun = TrainingRun()) -> None:
        """_summary_

        Args:
            env (_type_): A OpenAI Gym like environment
            learning (bool, optional): Switch on whether the agent will update the policy parameters or not. Defaults to True.
            temperature (int, optional): Soft-actor critic temperature. This is proportional to how much you want the policy to explore. Defaults to 2.
            load_actor_file (_type_, optional): pt file containing actor network which will be loaded. Defaults to None.
            num_runs (int, optional): number of runs you want the system to run for. Defaults to 1000.
        """

        # Initialise random seeds
        random_seed = np.random.randint(10000)
        torch.manual_seed(random_seed)
        np.random.seed(seed=random_seed)

        # Get current time for logging
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # Initialsie this log entry
        with open('RLAgentLog.txt', 'a') as f:
            line = "RLAgent run at :" + date_time_str + f"\n\tRandom seed: {random_seed}\n\n"
            f.write(line)

        # Set the device
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'

        # Save the environment
        self.env = env
        self.config = config

        # Get the shape of the state space
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Set the learning paraameters
        self.learning = learning
        self.batch_size= self.config.batch_size
        self.tau = self.config.tau
        self.max_steps = max_steps

        # Initialise the neural networks
        self._initialise_networks()
        self.all_networks = self._find_all_networks()

        # Set up save directory
        self.save_dir = self.config.save_directory

        self.gamma = self.config.gamma
        self.clip_param = 0.2
        self.best_score = 0

        if load_networks_dir:
            self.load_everything(load_networks_dir)

        self.scores = []
        self.smoothed_scores = []

        if not self.config.priority_mem:
            self.replay = ReplayBuffer(self.config.buffer_size, self.state_dim, self.action_dim)
        else:
            self.replay = ReplayBuffer(self.config.buffer_size, self.state_dim, self.action_dim, prioritise=self.config.priority_mem,
                                       policy_net=self.agent.actor_network, critic_nets=(self.critic_1, self.critic_2),
                                       target_net=self.target_value_net, gamma=self.gamma)
            
        # ---- Set up plotting -----
        self.plot_scores = True
        if self.plot_scores:
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots()
            scores = []  # List to store scores
            self.all_data_line, = self.ax.plot(scores, color='#D3D3D3', linewidth=2, label='Running mean')
            self.line, = self.ax.plot(self.smoothed_scores, color='#1E90FF', linewidth=2.5, label='Last 10')
            plt.xlabel('Runs')
            plt.ylabel('Score')
            plt.title('RL Algorithm Score Over Time')

        self.counter = 0
        self.iter_counter = 0

    
    # ------ RL METHODS -----
    @abstractmethod
    def _initialise_networks(self):
        """
        Implement initialisation of the function approximators being used for this algorithm
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_action(self, state: np.array):
        """
        Implement state to action function 
        Args:
            agent (Agent): _description_
            state (np.array): _description_

        """
        raise NotImplementedError
    
    @abstractmethod
    def learn(self):
        """

        Implement a way to learn using data from the replay buffer

        """
        raise NotImplementedError

    def explore_one_episode(self, learn: bool = None):
        """

        Explore one episode of the game until completion. The SARS' will be saved to the memory. If learning is set to true
        the function will be optimised too.

        Args:
            learn (bool, optional): Option to override the learning parameter set at initialisation. Defaults to None.
        """

        # Reset environment to start state
        seed = np.random.randint(0, 1000)
        state, _ = self.env.reset(seed=seed)
        score = 0

        # Create a temporary memory.
        #   All data is saved at the end of the episode. I made this change when I added prioritised experience
        #   replay as running the Q function on one batch per episode was far faster than per experience
        states= np.zeros((self.max_steps, self.state_dim), dtype=np.float32)
        new_states = np.zeros((self.max_steps, self.state_dim), dtype=np.float32)
        actions = np.zeros((self.max_steps, self.action_dim), dtype=np.float32)
        rewards = np.zeros((self.max_steps), dtype=np.float32)
        terminals = np.zeros(self.max_steps, dtype=np.bool_)

        # Count number of steps in episode
        for t in count():

            # Choose an action
            action = self.get_action(state)

            # Update environment
            numpy_action = action.cpu().detach().numpy()

            # Step the environment
            new_state, reward, done, truncated, _ = self.env.step(numpy_action)

            if t==0:
                self.first_action = numpy_action

            # Save the state transition in the temperory memory
            states[t] = state
            new_states[t] = new_state
            actions[t] = numpy_action
            rewards[t] = reward
            terminals[t] = done
 
            done = done or truncated

            if t==(self.max_steps-1) or done:
                truncated = True
            else:
                done = False

            # This gives the user the option to modify OpenAI's reward function if they so wish
            #   By default it just returns the values
            #   Check modified_agents/bipedal_walker for an example
            reward, done = self.modify_reward(reward, done)
            score += reward

            # If the agent is learning, run a step of learning.
            if self.learning:
                self.learn(self.agent)

            if done or truncated:
                # TODO: check this is actually needed?
                state = self.env.reset()
                break
            
            # Update the state and iteration counter
            state = new_state
            self.iter_counter += 1
        
        # Print an updatae for the user
        print(f"RUN: {self.counter}  Score: {score: .3f} \tTemperature: {self.temperature.item():.3e} \tLast std: {self.agent.actor_network.std.mean().item(): .3f}")
        print(f"\tCurrent entropy: {self.current_entropy: .3e} \t First action: {np.round(self.first_action, 3)} \n\n")

        # Save SARS to memory
        steps = t + 1 # Send the user memory space to the replay buffer
        self.replay.store_transition(steps, states[:steps], actions[:steps], rewards[:steps], new_states[:steps], terminals[:steps])

        # Step the learning rates
        if self.learning:
            self.step_learning_rates()

        
        # ---- PLOTTING -----

        self.scores.append(score)
        smoothing_range = 50

        if len(self.scores) > smoothing_range:
            self.smoothed_scores.append(np.mean(self.scores[-smoothing_range:]))
        
        if self.plot_scores:
            x_axis_data = np.arange(len(self.scores))
            # self.all_data_line.set_ydata(self.scores)
            # self.all_data_line.set_xdata(x_axis_data)
            if self.counter > smoothing_range:
                self.line.set_ydata(self.smoothed_scores)
                self.line.set_xdata(x_axis_data[smoothing_range:])
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        if self.counter%100==0:
            name_str = f"epoch_{self.counter}_"
            self.save_everything(folder='waypoints', tag=name_str)

        if score > self.best_score:
            self.save_everything()
            score = self.best_score
        
        self.counter += 1


    def modify_reward(self, reward: float, done: bool):
        """ 
        Overwrite this function to modify reward or completion functions
        """
        return reward, done

    def conservative_network_update(self, network: nn.Module, target_network: nn.Module, tau: float =None):
        """

        This will perform conversative update on two networks. This is where the target network is set to be a weighted 
        average is taken between two networks,with tau as the weighting. This is normally a small value such that 
        the target network is only updated very slowly.

        Args:
            network (nn.Module): in use network
            target_network (nn.Module): target network being used to update the in use network
            tau (float, optional): weighting off new network. Defaults to None.

        Returns:
            network (nn.Module): update in use network
            target_network (nn.Module): updated target network
        """

        if tau is None:
            tau = self.tau

        # Extract both network parameters
        learned_params = network.state_dict()
        target_params= target_network.state_dict()

        # Update target parameters to weighted average of the two
        for key in learned_params :
            target_params[key] = learned_params[key]*tau + target_params[key]*(1-tau)

        # Update target network (check this is needed and target params is not already a reference)
        target_network.load_state_dict(target_params)

        return network, target_network
    
    def step_learning_rates(self):

        # Below automatically finds all the neural networks and agents
        #   It will then step their learning rates
        #   User can overwrite this is needed
        [network.optimiser.step() for network in self.all_networks]

    def save_everything(self,*args, **kwargs):
        """ Save all the neural networks
        """
        [network.save_network(*args, **kwargs) for network in self.all_networks]

    def load_everything(self,*args, **kwargs):
        """ Load all the neural networks """
        [network.load_network(*args, **kwargs) for network in self.all_networks]

    def _find_all_networks(self):
        """

        I supplied a generic network template at RL/generic_network.py

        If this is used it will supply all the functionality required

        Returns:
            _type_: _description_
        """
        networks: List[GenericNetwork] = []
        for name, attribute in self.__dict__.items():
            if isinstance(attribute, nn.Module) and not isinstance(attribute, nn.modules.loss._Loss):
                networks.append(attribute)
            elif isinstance(attribute, Agent):
                networks.append(attribute.actor_network)
        return networks

    