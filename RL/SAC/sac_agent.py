import torch
from torch import optim, nn
import numpy as np

from RL.agent import RLAgent, Agent
from config import TrainingRun
from RL.SAC.networks import ActorNetwork, CriticNetwork, ValueNetwork

class SACAgent(RLAgent):

    def __init__(self, *args, temperature: float = None, **kwargs) -> None:

        # Initialise RL agent
        super().__init__(*args, **kwargs)
        
        # Set SAC specific hyperparameters 
        self.temperature = temperature

        # Set up the learnt temperature
        if self.temperature is None:

            self.auto_temp = True
            # Set target entropy
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            print(f'Target entropy set to: {self.target_entropy}')
            self.log_temperature = torch.tensor([self.config.start_log_temp], requires_grad=True, device=self.device)
            self.temperature_optim = optim.Adam([self.log_temperature], lr=self.config.temp_lr)

            self.temperature = self.log_temperature.exp()

        else:
            self.auto_temp = False
            self.temperature = torch.tensor(temperature)

        self.current_entropy = 0


    def _initialise_networks(self):

        # ------ Initialise networks --------
        self.agent = Agent(ActorNetwork(state_dim=self.state_dim, action_dim=self.action_dim, name="actor", min_action=self.env.action_space.low,
                                         max_action= self.env.action_space.high))

        # Two critic networks for stability
        self.critic_1 = CriticNetwork(state_dim=self.state_dim, action_dim=self.action_dim, name="critic_1")
        self.critic_2 = CriticNetwork(state_dim=self.state_dim, action_dim=self.action_dim, name="critic_2")

        # Value network
        self.value_net = ValueNetwork(state_dim=self.state_dim, name='value')
        self.target_value_net = ValueNetwork(state_dim=self.state_dim, name='target')
        self.val_loss = nn.MSELoss()

        self.value_net, self.target_value_net = self.conservative_network_update(self.value_net, self.target_value_net)

    def get_action(self, state: np.array):
        # Sample action from actor network
        action, _ = self.agent.actor_network.sample_actions(torch.tensor(state, dtype=torch.float32, device=self.device))
        return action
    
    def learn(self, agent):
        """

        Soft actor critic learning function

        Args:
            agent (_type_): The agent who is learning
        """
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