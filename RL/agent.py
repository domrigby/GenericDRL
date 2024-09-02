from RL.networks import ActorNetwork, CriticNetwork, ValueNetwork
from RL.memory import ReplayBuffer
import torch
from torch import nn, optim

from itertools import count
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

from config import TrainingRun

class Agent:
    def __init__(self, network):
        # Place holder for future potential multi-agent
        self.actor_network = network

class RLAgent:

    def __init__(self, env, temperature=None, learning=True, load_networks_dir=None, max_steps=1000, config=None) -> None:
        """_summary_

        Args:
            env (_type_): A OpenAI Gym like environment
            learning (bool, optional): Switch on whether the agent will update the policy parameters or not. Defaults to True.
            temperature (int, optional): Soft-actor critic temperature. This is proportional to how much you want the policy to explore. Defaults to 2.
            load_actor_file (_type_, optional): pt file containing actor network which will be loaded. Defaults to None.
            num_runs (int, optional): number of runs you want the system to run for. Defaults to 1000.
        """

        if config is None:
            self.config = TrainingRun()

        random_seed = np.random.randint(10000)
        torch.manual_seed(random_seed)
        np.random.seed(seed=random_seed)

        # Get current time for logging
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        with open('RLAgentLog.txt', 'a') as f:
            line = "RLAgent run at :" + date_time_str + f"\n\tRandom seed: {random_seed}\n\n"
            f.write(line)

        self.env = env

        # Get the shape of the state space
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Set the learning paraameters
        print(learning)
        self.learning = learning
        self.batch_size= self.config.batch_size
        self.tau = self.config.tau
        self.max_steps = max_steps

        # Set up save directory
        self.save_dir = self.config.save_directory

        self.gamma = self.config.gamma
        self.clip_param = 0.2

        self.best_score = 0

        # TODO: add action dims and state dims to game env
        # TODO: expand for multipy agents

        self.agent = Agent(ActorNetwork(state_dim=self.state_dim, action_dim=self.action_dim, name="actor", min_action=env.action_space.low,
                                         max_action= env.action_space.high))

        # Two critic networks for stability
        self.critic_1 = CriticNetwork(state_dim=self.state_dim, action_dim=self.action_dim, name="critic_1")
        self.critic_2 = CriticNetwork(state_dim=self.state_dim, action_dim=self.action_dim, name="critic_2")

        # Value network
        self.value_net = ValueNetwork(state_dim=self.state_dim, name='value')
        self.target_value_net = ValueNetwork(state_dim=self.state_dim, name='target')
        self.val_loss = nn.MSELoss()

        self.value_net, self.target_value_net = self.conservative_network_update(self.value_net, self.target_value_net)

        if load_networks_dir:
            self.load_everything(load_networks_dir)

        self.scores = []
        self.smoothed_scores = []

        self.plot_scores = True

        if not self.config.priority_mem:
            self.replay = ReplayBuffer(self.config.buffer_size, self.state_dim, self.action_dim)
        else:
            self.replay = ReplayBuffer(self.config.buffer_size, self.state_dim, self.action_dim, prioritise=self.config.priority_mem,
                                       policy_net=self.agent.actor_network, critic_nets=(self.critic_1, self.critic_2),
                                       target_net=self.target_value_net, gamma=self.gamma)

        if temperature is None:

            self.auto_temp = True
            # Set target entropy
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.critic_1.device)).item()
            print(f'Target entropy set to: {self.target_entropy}')
            self.log_temperature = torch.tensor([self.config.start_log_temp], requires_grad=True, device=self.critic_1.device)
            self.temperature_optim = optim.Adam([self.log_temperature], lr=self.config.temp_lr)

            self.temperature = self.log_temperature.exp()

        else:
            self.auto_temp = False
            self.temperature = torch.tensor(temperature)

        self.current_entropy = 0

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
    
    def get_action(self, agent, game_state):
        # Sample action from actor network
        action, _ = agent.actor_network.sample_actions(torch.tensor(game_state, dtype=torch.float32, device=agent.actor_network.device))
        return action.cpu().numpy()

    def explore_one_episode(self):
        
        self.update_hyperparameters()

        # Reset environment to start state
        seed = np.random.randint(0, 1000)
        state, _ = self.env.reset(seed=seed)

        score = 0

        states= np.zeros((self.max_steps, self.state_dim), dtype=np.float32)
        new_states = np.zeros((self.max_steps, self.state_dim), dtype=np.float32)
        actions = np.zeros((self.max_steps, self.action_dim), dtype=np.float32)
        rewards = np.zeros((self.max_steps), dtype=np.float32)
        terminals = np.zeros(self.max_steps, dtype=np.bool_)

        # Count number of steps in episode
        for t in count():

            # Choose an action
            action, log_probs = self.agent.actor_network.sample_actions(state)

            # Update environment
            numpy_action = action.cpu().detach().numpy()
            log_probs = log_probs.cpu().detach().numpy()

            new_state, reward, done, truncated, _ = self.env.step(numpy_action)

            if t==0:
                self.first_action = numpy_action

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

            reward, done = self.modify_reward(reward, done)
            score += reward

            if self.learning:
                self.learn(self.agent)

            if done or truncated:
                state = self.env.reset()
                break
            
            state = new_state
            self.iter_counter += 1

        print(f"RUN: {self.counter}  Score: {score: .3f} \tTemperature: {self.temperature.item():.3e} \tLast std: {self.agent.actor_network.std.mean().item(): .3f}")
        print(f"\tCurrent entropy: {self.current_entropy: .3e} \t First action: {np.round(self.first_action, 3)} \n\n")

        # Save SARS to memory
        steps = t + 1
        self.replay.store_transition(steps, states[:steps], actions[:steps], rewards[:steps], new_states[:steps], terminals[:steps])

        # Step the learning rates
        self.step_learning_rates()

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

    def modify_reward(self, reward, done):
        """ 
        Overwrite this function to modify reward or completion functions
        """
        return reward, done
    
    def update_hyperparameters(self):
        pass

    def learn(self, agent):
        if self.replay.write_num < max(self.batch_size, self.config.learning_start):
            return
        
        # Retrieve sars from buffer
        state, action, reward, new_state, done, old_log_probs = \
                self.replay.sample_buffer(self.batch_size)

        #  ---------------------------- VALUE FUNCTION UPDATE ----------------------------------

        # Get the value of the current state
        value = self.value_net(state).view(-1) # values of the current states
        
        # Get an action to do at the current state and tehn criticise it (get the Q value)
        critic_val, log_probs = self.criticise(agent, state)

        # The value of a state in SAC not only includes expectation of value after action taken,
        # sampled from the policy, but also the entropy of the policy
        self.value_net.optimiser.zero_grad()
        value_target = critic_val - self.temperature*log_probs

        # The error of the value function is the difference between the predicted value (value) 
        # and the observed value at this state (value_target)
        value_loss = 0.5*self.val_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value_net.optimiser.step()

        # ---------------------------- POLICY NETWORK UPDATE ----------------------------------

        if self.iter_counter%self.config.policy_train_freq==0:

            # Get an action to do at the current state and tehn criticise it (get the Q value)
            critic_val, log_probs = self.criticise(agent, state, reparam=True)
            
            # Actor loss is log probs + negative critics value...
            #   ... why? If the actor minimises both of these it will max entropy and reward, soft and hard
            actor_loss = ((self.temperature*log_probs) - critic_val).mean()
            agent.actor_network.optimiser.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor_network.optimiser.step()

        # ---------------------------- CRITIC NETWORK UPDATE ----------------------------------

        # Critic networks are updated in usual Deep Q learning method
        target_values = self.target_value_net(new_state).view(-1) 

        # Set all final states to have a value function of 0, as this is in the definition of the value function
        target_values[done] = 0.0 # def of value func

        self.critic_1.optimiser.zero_grad()
        self.critic_2.optimiser.zero_grad()

        # Update in classic Q function method
        q_hat = reward + self.gamma*target_values
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * self.val_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * self.val_loss(q2_old_policy, q_hat)

        if self.config.priority_mem:
            q1_temp_error = q1_old_policy - q_hat
            q2_temp_error = q2_old_policy - q_hat
            mean_temp_error = q1_temp_error + q2_temp_error / 2.0
            self.replay.reassign_priority(mean_temp_error)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimiser.step()
        self.critic_2.optimiser.step()

        # Slowly update V target
        self.value_net, self.target_value_net = self.conservative_network_update(self.value_net, self.target_value_net)
        
        if self.auto_temp and self.iter_counter%self.config.policy_train_freq==0:

            with torch.no_grad():
                 _, log_probs = agent.actor_network.sample_actions(state)

            # Temperature loss: the temperature * the negative log entropties minus the target entropy
            temperature_loss = -(self.log_temperature.exp() * (log_probs+self.target_entropy).detach()).mean()

            self.temperature_optim.zero_grad()
            temperature_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_temperature], max_norm=1)
            self.temperature_optim.step()

            self.temperature = self.log_temperature.exp()

            # Current entropy is printing purposes only
            self.current_entropy = -(log_probs).mean().item()

    def criticise(self, agent, states, reparam=False):
        """
        Criticise function will generate an action in the current state and then criticise it using a Q function
        """

        # Sample the stochastic actor distribution for this state and get the logarithm of the prob.density
        # The log of the probabilities is there for future entropy calculation
        actions, log_probs = agent.actor_network.sample_actions(states, reparameterize=reparam)
        log_probs = log_probs.view(-1)

        # Get value of performing this action in this state (CRITICISE!)
        q1_of_new_pol = self.critic_1.forward(states, actions)
        q2_of_new_pol = self.critic_2.forward(states, actions)

        # The minimum is taken for stability reasons
        critic_val = torch.min(q1_of_new_pol, q2_of_new_pol)
        critic_val= critic_val.view(-1)

        return critic_val, log_probs

    def conservative_network_update(self, network, target_network, tau=None):

        if tau is None:
            tau = self.tau

        learned_params = network.state_dict()
        target_params= target_network.state_dict()

        for key in learned_params :
            target_params[key] = learned_params[key]*tau + target_params[key]*(1-tau)

        target_network.load_state_dict(target_params)

        return network, target_network

    def compute_advantages(self, rewards, values, next_values, dones):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages
    
    def step_learning_rates(self):
        # Step all learning rates
        self.value_net.scheduler.step()
        self.critic_1.scheduler.step()
        self.critic_2.scheduler.step()
        self.agent.actor_network.scheduler.step()

    def save_everything(self,*args, **kwargs):
        # find all networks... not loss
        self.agent.actor_network.save_network(*args, **kwargs)
        for name, attribute in self.__dict__.items():
            if isinstance(attribute, nn.Module) and not isinstance(attribute, nn.modules.loss._Loss):
                attribute.save_network(*args, **kwargs)

    def load_everything(self,*args, **kwargs):
        self.agent.actor_network.load_network(*args)
        for name, attribute in self.__dict__.items():
            if isinstance(attribute, nn.Module) and not isinstance(attribute, nn.modules.loss._Loss):
                attribute.load_network(*args, **kwargs)

    def animate(self):
        plt.cla()  # Clear the current axe
        plt.plot(np.convolve(self.scores), np.ones((10)/10))

    