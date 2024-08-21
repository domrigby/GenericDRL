from RL.agent import RLAgent

import gymnasium as gym
from modified_agents.bipedal_walker_agent import BipedalRLAgent

from config import TrainingRun

config_file = TrainingRun()

env = gym.make("BipedalWalker-v3", render_mode="human")

agent = BipedalRLAgent(env, max_steps=config_file.max_steps, temperature=config_file.Temperature) #, load_actor_file=True) #, learning=False, load_actor_file=True)

for i in range(100000):
    agent.explore_one_episode()
    #TODO: finish integrating of the sensors
