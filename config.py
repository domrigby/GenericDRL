class TrainingRun:
    def __init__(self) -> None:

        # Choose an OpenAI Gym environment
        self.Environment = "BipedalWalker-v3"

        # If you wish to load a set of networks saved with the default naming formmat
        # name the directory here
        self.learning = False
        self.dir_to_load = "working_networks/bipedal_walker"
        
        # The number of steps after which the episode will end
        self.max_steps = 5000

        # Set the temperature constant...
        #   Set to None if you wish to have an automatically adjusted temperature
        self.Temperature = None

        # Set up learning parameters
        self.learning = True

        # Set policy and critic learning
        self.q_learning_rate = 0.001
        self.policy_learning_rate = 0.0003

        # Set rate of conservatice update
        self.tau = 0.005

        self.batch_size = 256
        self.gamma = 0.99

        self.save_directory = "saved_networks"

        self.buffer_size = int(1e7)
        self.priority_mem = False

        self.start_log_temp = 1.0
        self.temp_lr = 0.0001

        self.learning_start = 5e2
        self.policy_train_freq = 2