import torch
from RL.agent import RLAgent
from config import TrainingRun

class SACAgent(RLAgent):

    def __init__(self, env, temperature: float = None, learning: bool = True, load_networks_dir: str = None, max_steps: int = 1000, config: TrainingRun = ...) -> None:
        super().__init__(env, temperature, learning, load_networks_dir, max_steps, config)