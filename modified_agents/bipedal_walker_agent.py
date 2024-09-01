from RL.agent import RLAgent

import gymnasium as gym
import numpy as np

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

        if done:
            return reward, done
        

        if not self.mod_called:
            self.furthest_right = CurrentPosition(self.env)
            self.last_position = CurrentPosition(self.env)
            self.mod_called = True
            return 0.0, False

        new_position = CurrentPosition(self.env)

        # if new_position.x > self.furthest_right.x:
        #     reward= 10.0*(new_position.x - self.furthest_right.x)
        #     self.furthest_right = new_position
        # else:
        #     reward = 0

        if np.linalg.norm(new_position - self.last_position) < 0.015:
            self.stuck_count += 1
            if self.stuck_count > 500:
                done = True
                reward -= 100
        else:
            self.stuck_count = 0

        self.last_position = new_position

        return reward, done
