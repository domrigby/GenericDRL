class TrainingRun:
    def __init__(self) -> None:

        self.Environment = "BipedalWalker-v3"

        self.dir_to_load = "saved_c"

        self.MaxSteps = 300

        self.Temperature = 0.25

        # Set up learning parameters
        self.learning = True
        self.learning_rate = 0.00025
        self.batch_size = 256
        self.tau = 0.005
        self.gamma = 0.99

        self.buffer_size = 25000
