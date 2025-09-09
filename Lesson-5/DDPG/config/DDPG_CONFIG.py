DDPG_CONFIG = {
    "num_episodes": 5000,
    "initial_noise": 0.1,
    "noise_decay": 0.999,
    "min_noise": 0.01,
    "memory_size": 1e6,
    "batch_size": 128,
    "tau": 5e-2,
    "gamma": 0.99,
    "evaluation_period": 50,
    "evaluation_episodes": 10,
}
