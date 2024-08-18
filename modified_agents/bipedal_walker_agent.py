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


class BipedalRLAgent(RLAgent):

    def __init__(self, env, temperature=None, learning=True, load_actor_file=None, max_steps=1000) -> None:
        self.stuck_count = 0
        super().__init__(env, temperature, learning, load_actor_file, max_steps)

    def explore_one_episode(self):
        self.stuck_count = 0
        self.mod_called = False
        return super().explore_one_episode()

    def modify_reward(self, reward, done):

        # if done:
        #     reward = 0
        #     return reward, done
        

        # if not self.mod_called:
        #     self.current_position = CurrentPosition(self.env)
        #     self.mod_called = True

        # new_position = CurrentPosition(self.env)

        # if new_position.x > self.current_position.x:
        #     reward= (new_position.x - self.current_position.x)
        #     self.current_position = new_position
        # else:
            # reward = 0
        

        return reward, done
