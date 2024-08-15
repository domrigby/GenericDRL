from RL.agent import RLAgent

import gymnasium as gym

class CurrentPosition():
    def __init__(self, x=0, y=0) -> None:
        self.x = 0
        self.y = 0
        self.been_set = False

    def set(self,x,y):
        self.x = x
        self.y = y
        self.been_set = True


class ModifedRLAgent(RLAgent):

    def __init__(self, env, learning=True, temperature=2, load_actor_file=None, max_steps=1000) -> None:
        self.current_position = 0
        super().__init__(env, learning, temperature, load_actor_file, max_steps)

    def explore_one_episode(self):
        self.current_position = CurrentPosition()
        return super().explore_one_episode()

    def modify_reward(self, reward):
        if not self.current_position.been_set:
            self.current_position.set(self.env.hull.position[0], self.env.hull.position[1])
            return 0

        distance_x = (self.env.hull.position[0] - self.current_position.x)
        distance_y = (self.env.hull.position[1] - self.current_position.y)

        reward = distance_x

        self.current_position.set(self.env.hull.position[0], self.env.hull.position[1])

        return reward
