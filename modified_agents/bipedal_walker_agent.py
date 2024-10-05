from RL.agent import RLAgent

import gymnasium as gym
import numpy as np

from RL.networks import ActorNetwork

class CurrentPosition():
    def __init__(self, env) -> None:
        self.x = env.hull.position.x
        self.y = env.hull.position.y
        self.been_set = False

    def set(self,env):
        self.x = env.hull.position.x
        self.y = env.hull.position.y
        self.been_set = True

    def __sub__(self, other):
        if isinstance(other, CurrentPosition):
            return np.array([self.x - other.x, self.y - other.y])
        return NotImplemented


class BipedalRLAgent(RLAgent):

    def __init__(self, *args, **kwargs) -> None:
        self.stuck_count = 0
        super().__init__(*args, **kwargs)

    def explore_one_episode(self):
        self.stuck_count = 0
        self.mod_called = False
        return super().explore_one_episode()

    def modify_reward(self, reward, done):

        # if done:
        #     return reward, done
        

        # if not self.mod_called:
        #     self.furthest_right = CurrentPosition(self.env)
        #     self.last_position = CurrentPosition(self.env)
        #     self.mod_called = True
        #     return 0.0, False

        # new_position = CurrentPosition(self.env)

        # # if new_position.x > self.furthest_right.x:
        # #     reward= 10.0*(new_position.x - self.furthest_right.x)
        # #     self.furthest_right = new_position
        # # else:
        # #     reward = 0

        # if np.linalg.norm(new_position - self.last_position) < 0.015:
        #     self.stuck_count += 1
        #     if self.stuck_count > 500:
        #         done = True
        #         reward -= 100
        # else:
        #     self.stuck_count = 0

        # self.last_position = new_position

        return reward, done


from torch import nn
import torch
class BWActor(ActorNetwork):

    def __init__(self, state_dim, action_dim, name="", min_action=1, max_action=1) -> None:
        super().__init__(state_dim, action_dim, name, min_action, max_action)
        self.kin_fc1 = nn.Linear(state_dim-10, 256)
        self.lidar_net = nn.Sequential(nn.Conv1d(1,32,padding=0,stride=1), 
                                    nn.Conv1d(32,8,padding=0,stride=1),
                                    nn.Flatten(),
                                    nn.Linear(80,10))

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        dynamics = state[:-10]
        lidar_data= state[-10:]

        x = self.act(self.kin_fc1(state))
        x = self.act(self.kin_fc2(x))

        lidar_feat = self.lidar_net(lidar_data)

        x = torch.cat([x, lidar_feat])

        # Get mean and std 
        mean = self.mean(x)
        log_std = self.log_sigma(x)

        log_std = torch.tanh(log_std)
        log_std = -5 + 0.5 * 7 * (log_std + 1)

        std = log_std.exp()

        self.std = std

        return mean, std