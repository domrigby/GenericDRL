from RL.networks import ActorNetwork, CriticNetwork, ValueNetwork
from RL.memory import ReplayBuffer
import torch
from torch import nn

from itertools import count

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

class Agent:
    def __init__(self, network):
        # Place holder for future potential multi-agent
        self.actor_network = network

class RLAgent:

    def __init__(self, env, learning=True, temperature=2, load_actor_file=None, max_steps=1000) -> None:
        """_summary_

        Args:
            env (_type_): A OpenAI Gym like environment
            learning (bool, optional): Switch on whether the agent will update the policy parameters or not. Defaults to True.
            temperature (int, optional): Soft-actor critic temperature. This is proportional to how much you want the policy to explore. Defaults to 2.
            load_actor_file (_type_, optional): pt file containing actor network which will be loaded. Defaults to None.
            num_runs (int, optional): number of runs you want the system to run for. Defaults to 1000.
        """
        self.env = env

        # Get the shape of the state space
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # Temperature is the weighting of the entropy in the loss function
        self.temperature = temperature

        # Set the learning paraameters
        self.learning = learning
        self.batch_size= 128
        self.tau = 0.005
        self.max_steps = max_steps

        self.gamma = 0.99

        self.replay = ReplayBuffer(100000, state_dim, action_dim)

        # TODO: add action dims and state dims to game env
        # TODO: expand for multipy agents

        self.agent = Agent(ActorNetwork(state_dim=state_dim, action_dim=action_dim, name="actor", max_action= env.action_space.high))

        if load_actor_file:
            self.load_everything()

        # Two critic networks for stability
        self.critic_1 = CriticNetwork(state_dim=state_dim, action_dim=action_dim, name="critic_1")
        self.critic_2 = CriticNetwork(state_dim=state_dim, action_dim=action_dim, name="critic_2")

        # Value network
        self.value_net = ValueNetwork(state_dim=state_dim, name='value')
        self.target_value_net = ValueNetwork(state_dim=state_dim, name='target')
        self.val_loss = nn.MSELoss()

        self.conversative_value_update(tau=1)

        self.scores = []
        self.smoothed_scores = []

        self.plot_scores = True

        if self.plot_scores:
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots()
            scores = []  # List to store scores
            self.line, = self.ax.plot(scores)
            plt.xlabel('Runs')
            plt.ylabel('Score')
            plt.title('RL Algorithm Score Over Time')

        self.counter = 0
    
    def get_action(self, agent, game_state):
        # Sample action from actor network
        action, _ = agent.actor_network.sample_actions(torch.tensor(game_state, dtype=torch.float32, device=agent.actor_network.device))
        return action.cpu().numpy()

    def explore_one_episode(self):

        # Reset environment to start state
        state, _ = self.env.reset()

        score = 0

        # Count number of steps in episode
        for t in count():

            # Choose an action
            action, _ = self.agent.actor_network(state)

            # Update environment
            numpy_action = action.cpu().detach().numpy()

            new_state, reward, done, truncated, _ = self.env.step(numpy_action)

            done = done or truncated

            if t==self.max_steps or done:
                truncated = True
            else:
                done = False

            reward = self.modify_reward(reward)
            score += reward

            # Save SARS to memory

            self.replay.store_transition(state, numpy_action, reward, new_state, done)

            if self.learning:
                self.learn(self.agent)

            if done or truncated:
                state = self.env.reset()
                break

            # print(f"Step {t} state value : {self.value_net(torch.tensor(state).cuda())[0]:.3f} {self.env.hull.position}")
            
            state = new_state

        self.scores.append(score)

        mean_length = 50
        if len(self.scores) > mean_length:
            self.smoothed_scores.append(np.mean(self.scores[-mean_length:]))
        
        if self.plot_scores:
            smoothing_range = 10
            if self.counter > smoothing_range:
                self.line.set_ydata(self.smoothed_scores)
                self.line.set_xdata(range(len(self.smoothed_scores)))
                self.ax.relim()
                self.ax.autoscale_view()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

        if self.counter%10==0:
            self.save_everything()
        
        self.counter += 1

    def modify_reward(self, reward):
        """ Function to modify reward if needed
        """
        return reward

    def learn(self, agent):

        if self.replay.mem_cntr < self.batch_size:
            return
        
        # Retrieve sars from buffer
        state, action, reward, new_state, done = \
                self.replay.sample_buffer(self.batch_size)
        
        # Create tensors from numpy arrays
        reward = torch.tensor(reward, dtype=torch.float).to(agent.actor_network.device)
        done = torch.tensor(done).to(agent.actor_network.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(agent.actor_network.device)
        state = torch.tensor(state, dtype=torch.float).to(agent.actor_network.device)
        action = torch.tensor(action, dtype=torch.float).to(agent.actor_network.device)


        #  ---------------------------- VALUE FUNCTION UPDATE ----------------------------------

        # Get the value of the current state
        value = self.value_net(state).view(-1) # values of the current states
        
        # Get an action to do at the current state and tehn criticise it (get the Q value)
        critic_val, log_probs = self.criticise(agent, state)

        # The value of a state in SAC not only includes expectation of value after action taken,
        # sampled from the policy, but also the entropy of the policy
        self.value_net.optimiser.zero_grad()
        value_target = critic_val - log_probs

        # The error of the value function is the difference between the predicted value (value) 
        # and the observed value at this state (value_target)
        value_loss = 0.5*self.val_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value_net.optimiser.step()

        # ---------------------------- POLICY NETWORK UPDATE ----------------------------------

        # Get an action to do at the current state and tehn criticise it (get the Q value)
        critic_val, log_probs = self.criticise(agent, state, reparam=True)

        # Actor loss is log probs + negative critics value...
        #   ... why? If the actor minimises both of these it will max entropy and reward, soft and hard
        actor_loss = log_probs - critic_val
        actor_loss = torch.mean(actor_loss)
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

        # Update inn classic Q function method
        q_hat = self.temperature*reward + self.gamma*target_values
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * self.val_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * self.val_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimiser.step()
        self.critic_2.optimiser.step()

        # Slowly update V target
        self.conversative_value_update()

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

    def conversative_value_update(self,tau=None):

        if tau is None:
            tau = self.tau

        V_target_params= self.target_value_net.state_dict()
        V_learned_params = self.value_net.state_dict()
        for key in V_learned_params :
            V_target_params[key] = V_learned_params[key]*tau + V_target_params[key]*(1-tau)
        self.target_value_net.load_state_dict(V_target_params)

    def save_everything(self):
        # find all networks... not loss
        for name, attribute in self.__dict__.items():
            if isinstance(attribute, nn.Module) and not isinstance(attribute, nn.modules.loss._Loss):
                attribute.save_network()

    def load_everything(self):
        for name, attribute in self.__dict__.items():
            if isinstance(attribute, nn.Module) and not isinstance(attribute, nn.modules.loss._Loss):
                attribute.load_network()

    def animate(self):
        plt.cla()  # Clear the current axe
        print("HEre")
        plt.plot(np.convolve(self.scores), np.ones((10)/10))

    