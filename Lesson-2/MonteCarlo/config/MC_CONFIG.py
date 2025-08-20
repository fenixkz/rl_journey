MC_CONFIG = {
    "gamma": 0.99,
    "alpha": 0.01,
    "num_episodes": 1e6,
    "start_epsilon": 1.0,
    # Monte-Carlo needs long exploration
    # to fight variance
    "min_epsilon": 0.1,
    # Number of episodes to decay from start_epsilon to min_epsilon
    "decay_episodes": 6e5,
    "num_bins": 200,
    "evaluation_period": 5000,
    "evaluation_episodes": 10,
}
