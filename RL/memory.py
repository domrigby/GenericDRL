# CREDIT FOR THIS MEMORY CLASS GOES TO PHIL TABOR OFF YOUTUBE 

import numpy as np 
import torch

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, torch_device='cuda',
                 prioritise=False, critic_nets=None, policy_net=None, 
                 target_net =None, gamma=None):
        self.mem_size = max_size
        self.write_num = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        self.priority = np.zeros(self.mem_size)

        memory_usage = self.state_memory.nbytes + self.new_state_memory.nbytes + \
                       self.action_memory.nbytes + self.reward_memory.nbytes +\
                       self.terminal_memory.nbytes + self.priority.nbytes
        

        print(f"Buffer is using {'%.2e' % memory_usage} bytes")

        self.prioritise = prioritise
        self.device = torch_device

        if prioritise:
            self.critic_nets = critic_nets
            self.gamma = gamma
            self.policy_net = policy_net
            self.target_net = target_net

            self.last_calc_len = 0
            self.probs = 0

        self.total = 0

    def store_transition(self, num_done, state, action, reward, new_state, done):
        """
        Stores sars tuples. Starts overwriting the beginning ones when full
        """

        index = self.write_num % self.mem_size
        index = min(self.mem_size-num_done, index)

        # if (index+num_done) > self.mem_size:
        #     index = min(self.mem_size-num_done-1, index)

        self.state_memory[index:index+num_done] = state[:num_done]
        self.new_state_memory[index:index+num_done] = new_state[:num_done]
        self.action_memory[index:index+num_done] = action[:num_done]
        self.reward_memory[index:index+num_done] = reward[:num_done]
        self.terminal_memory[index:index+num_done] = done[:num_done]

        if self.prioritise:
            self.priority[index:index+num_done] = self.calculate_temp_error(state[:num_done], 
                                    action[:num_done], reward[:num_done], new_state[:num_done])

        self.write_num += num_done
        self.total += num_done

        if self.prioritise and self.write_num > self.mem_size:
            
            # This is to start overwriting the bad experience first.

            print("\n\nRESHUFFLE\n\n")

            sorted_indices = np.argsort(self.priority)

            self.state_memory = self.state_memory[sorted_indices]
            self.new_state_memory = self.new_state_memory[sorted_indices]
            self.action_memory = self.action_memory[sorted_indices]
            self.reward_memory = self.reward_memory[sorted_indices]
            self.terminal_memory = self.terminal_memory[sorted_indices]
            self.priority = self.priority[sorted_indices]

            self.write_num -= self.mem_size
            self.write_num = max(self.write_num, 1)

    def sample_buffer(self, batch_size):

        max_mem = min(self.total, self.mem_size)

        if self.prioritise:
            if self.write_num != self.last_calc_len:
                total_p = sum(self.priority[:max_mem])
                max_p = 5*total_p/max_mem
                min_p = (total_p/max_mem)/5
                new_ps =  np.clip(self.priority[:max_mem],a_min=min_p, a_max=max_p)
                self.probs = new_ps/sum(new_ps)
                self.last_calc_len = self.write_num
            
            self.current_batch = np.random.choice(max_mem, batch_size, p=self.probs)
        else:
            self.current_batch= np.random.choice(max_mem, batch_size)

        states = self.state_memory[self.current_batch]
        new_states = self.new_state_memory[self.current_batch]
        actions = self.action_memory[self.current_batch]
        rewards = self.reward_memory[self.current_batch]
        dones = self.terminal_memory[self.current_batch]

         # Create tensors from numpy arrays
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        new_states = torch.tensor(new_states, dtype=torch.float).to(self.device)
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)

        return states, actions, rewards, new_states, dones, None
    

    def calculate_temp_error(self, state, action, reward, new_state):
        
        delta = np.zeros((state.shape[0], len(self.critic_nets)))

        with torch.no_grad():
            for i, net in enumerate(self.critic_nets):

                state = torch.tensor(state).to(self.device)
                new_state = torch.tensor(new_state).to(self.device)
                action = torch.tensor(action).to(self.device)
                reward = torch.tensor(reward).to(self.device)

                delta[:,i] = abs(((reward.unsqueeze(-1) + self.gamma*self.target_net(new_state.detach())) 
                                - net(state.detach(), action.detach())).squeeze(-1).cpu().numpy())
        return np.mean(delta, axis=1)
    
    def reassign_priority(self, temp_errors):
        self.priority[self.current_batch] = np.abs(temp_errors.detach().cpu().numpy())