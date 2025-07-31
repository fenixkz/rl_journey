# --- Default algorithm hyperparameters ---
# These are common values from DQN literature, used for most games unless specified otherwise.
DEFAULT_ATARI_PARAMS = {
    # ----- COMMON PARAMS -----
    "max_steps": int(5e6), # Total number of steps to train
    "gamma": 0.99, # Gamma of TD update rule
    "memory_size": int(1e6), # Replay or PER total size
    "learning_starts": 5e4, # How much to samples to gather before the learning starts
    "batch_size": 32, # How many samples to use for gradient update, same as in Nature paper
    "lr": 2.5e-4, # Learning rate, same as in Nature paper
    "learning_freq": 2, # How many k-steps to update weights, same as in Nature paper
    "hidden_dim": 512, # Network width, same as in Nature paper
    # ----- TARGET NETWORK UPDATE PARAMS -----
    "hard_target_update": False, # Use either hard or soft update
    "target_update_freq": 10000, # Nature paper
    "tau": 0.005,
    # ----- EPSILON-GREEDY PARAMS -----
    "epsilon_decay_steps": int(1e6), # Total number of steps to decay linearly epsilon from max to min, same as in Nature paper
    "max_epsilon": 1, # Initial epsilon, same as in Nature paper
    "min_epsilon": 0.1, # Final epsilon, same as in Nature paper
    # ----- PRIORITIZED EXPERIENCE REPLAY PARAMS ----
    "alpha": 0.6, # Alpha of PER
    "beta_start": 0.4, # Starting Beta of PER
    "beta_final": 1, # Final Beta of PER
    "beta_increase_steps": int(1e5), # Total number of steps to increase beta from start to end
    # ----- N-STEP RETURN PARAMS -----
    "n_step_return": 3, # RAINBOW benchmark
}

DEFAULT_CLASSIC_PARAMS = {
    # ----- COMMON PARAMS -----
    "max_steps": int(3e5),
    "gamma": 0.99,
    "memory_size": int(5e4), # Use smaller size to dicard old samples because they can be harmful
    "learning_starts": int(1e2), # No need to wait long
    "batch_size": 32, 
    "lr": 5e-4, # We can use a higher lr
    "learning_freq": 1, # Box2D envs are fast, so no need to skip 
    "hidden_dim": 64, # Common value for small problems
    # ----- TARGET NETWORK UPDATE PARAMS -----
    "hard_target_update": False, # Use either hard or soft update
    "target_update_freq": 1000, # We can update much faster
    "tau": 0.005,
    # ----- EPSILON-GREEDY PARAMS -----
    "epsilon_decay_steps": int(1e4), # Quick exploration
    "max_epsilon": 1, 
    "min_epsilon": 0.02, # To be more greedy 
    # ----- PRIORITIZED EXPERIENCE REPLAY PARAMS ----
    "alpha": 0.6, # Alpha of PER
    "beta_start": 0.4, # Starting Beta of PER
    "beta_final": 1, # Final Beta of PER
    "beta_increase_steps": int(1e5), # Total number of steps to increase beta from start to end
    # ----- N-STEP RETURN PARAMS -----
    "n_step_return": 3, # RAINBOW benchmark
}


from dataclasses import dataclass

@dataclass
class AgentConfig:
    # ----- COMMON PARAMS -----
    max_steps: int = int(3e5)
    gamma: float = 0.99
    memory_size: int = int(1e4)
    learning_starts: int = int(1e3)
    batch_size: int = 64
    lr: float = 5e-4
    learning_freq: int = 1
    hidden_dim: int = 64
    # ----- TARGET NETWORK UPDATE PARAMS -----
    hard_target_update: bool = False
    target_update_freq: int = 500
    tau: float = 0.005
    # ----- EPSILON-GREEDY PARAMS -----
    epsilon_decay_steps: int = int(5e4)
    max_epsilon: float = 1.0
    min_epsilon: float = 0.02
    eps_decay: float = None  # This will be calculated in __post_init__
    # ----- PRIORITIZED EXPERIENCE REPLAY PARAMS ----
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_final: float = 1.0
    beta_increase_steps: int = int(1e5)
    # ----- N-STEP RETURN PARAMS -----
    n_step_return: int = 3
    # ----- OTHERS -----
    seed: int = 224
    # ----- SPECIFIC PARAMS TO DIFFERENTIATE DQN AGENTS ----
    memory: str = "rb" # Either replay buffer or PER
    dueling: bool = False # Dueling DQN requires different network archs

    def __post_init__(self):
        """Calculate eps_decay after initialization with the actual parameter values"""
        self.eps_decay = (self.max_epsilon - self.min_epsilon) / self.epsilon_decay_steps

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AgentConfig':
        """
        Create an AgentConfig instance from a dictionary configuration.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            AgentConfig instance with values from the dictionary
            
        Example:
            config = AgentConfig.from_dict(DEFAULT_ATARI_PARAMS)
            config = AgentConfig.from_dict(DEFAULT_CLASSIC_PARAMS)
        """
        # Create a copy of the input dict to avoid modifying the original
        params = config_dict.copy()
        
        # Convert values to appropriate types and handle int() expressions
        if 'max_steps' in params:
            params['max_steps'] = int(params['max_steps'])
        if 'memory_size' in params:
            params['memory_size'] = int(params['memory_size'])
        if 'learning_starts' in params:
            params['learning_starts'] = int(params['learning_starts'])
        if 'epsilon_decay_steps' in params:
            params['epsilon_decay_steps'] = int(params['epsilon_decay_steps'])
        if 'beta_increase_steps' in params:
            params['beta_increase_steps'] = int(params['beta_increase_steps'])
            
        # Filter out any keys that don't exist in the dataclass
        import inspect
        valid_fields = set(inspect.signature(cls).parameters.keys())
        filtered_params = {k: v for k, v in params.items() if k in valid_fields}
        
        return cls(**filtered_params)