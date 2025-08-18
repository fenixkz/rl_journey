ES_CONFIG = {
    "training_generations": 10000,
    "population_size": 50,
    "noise_std": 0.75,
    "hidden_dim": 64,
    "learning_rate": 3e-4,
    "use_vbn": True,
    "vbn_batch_size": 128,
    "l2_coeff": 1e-3,
    "normalization_mode": "default",
    "action_noise": 0.3,
    "evaluation_period": 100,
    "evaluation_episodes": 20,
}


ES_ATARI_CONFIG = {
    "training_generations": 5000,
    "population_size": 100,
    "noise_std": 0.05,
    "hidden_dim": 512,
    "learning_rate": 0.001,
    "use_vbn": True,
    "vbn_batch_size": 128,
    "l2_coeff": 5e-3,
    "normalization_mode": "rank",
    "action_noise": 0.1,
    "evaluation_period": 1000,
    "evaluation_episodes": 10,  # Atari envs are much slower
}
