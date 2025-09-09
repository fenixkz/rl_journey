TD3_CONFIG = {
    "num_steps": 1e6,
    "exploration_noise": 0.2,
    "memory_size": 1e6,
    "batch_size": 256,
    "tau": 5e-3,
    "gamma": 0.99,
    "evaluation_period": 5000,  # steps
    "evaluation_episodes": 10,
    "policy_noise": 0.2,
    "policy_noise_bound": 0.5,
    "num_envs": 16,
    "actor_update_rate": 2,
}
