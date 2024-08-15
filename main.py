from RL.agent import RLAgent

import gymnasium as gym

# env = gym.make("InvertedPendulum-v4", render_mode="human")
env = gym.make("BipedalWalker-v3", render_mode="human")

#env = gym.make("LunarLander-v2", continuous = True)

agent =  RLAgent(env, max_steps=150, temperature=5, learning=False, load_actor_file=True)

for i in range(100000):
    agent.explore_one_episode()
    #TODO: finish integrating of the sensors
