class TrainingRun:
    def __init__(self) -> None:

        # Choose an OpenAI Gym environment
        self.Environment = "BipedalWalker-v3"

        # If you wish to load a set of networks saved with the default naming formmat
        # name the directory here
        self.dir_to_load = None

        # The number of steps after which the episode will end
        self.max_steps = 1000

        # Set the temperature constant...
        #   Set to None if you wish to have an automatically adjusted temperature
        self.Temperature = None

        # Set up learning parameters
        self.learning = True
        self.q_learning_rate = 0.001
        self.policy_learning_rate = 0.0003
        self.batch_size = 256
        self.tau = 0.005
        self.gamma = 0.99

        self.save_directory = "saved_networks"

        self.buffer_size = 100000
        self.priority_mem = False

        self.learning_start = 5e2
        self.policy_train_freq = 2