import numpy as np
import random

class TestGame:

    action_space = np.array([1])
    observation_space = np.array([1])

    def __init__(self):
        self.target= 0.5