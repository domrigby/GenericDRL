from RL.SAC.sac_agent import SACAgent

import gymnasium as gym

from config import TrainingRun

config_file = TrainingRun()

#env = gym.make("LunarLander-v2", render_mode="human", continuous=True)

env = gym.make("BipedalWalker-v3", render_mode="human", hardcore=True)

#env = gym.make("InvertedDoublePendulum-v4", render_mode="human")
agent = SACAgent(env, max_steps=config_file.max_steps, temperature=config_file.Temperature, load_networks_dir=config_file.dir_to_load, learning=config_file.learning) 
for i in range(100000):
    agent.explore_one_episode()
    #TODO: finish integrating of the sensors
 