class TrainingRun:
    def __init__(self) -> None:

        self.Environment = "BipedalWalker-v3"

        self.dir_to_load = None

        self.max_steps = 300

        self.Temperature = None

        # Set up learning parameters
        self.learning = True
        self.learning_rate = 0.0003
        self.batch_size = 256
        self.tau = 0.005
        self.gamma = 0.99

        self.save_directory = "saved_networks"

        self.buffer_size = 150000
        self.priority_mem = False
