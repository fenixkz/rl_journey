SAC_CONFIG = {
    "num_steps": 1e6,
    "memory_size": 1e6,
    "batch_size": 256,
    "tau": 5e-3,
    "gamma": 0.99,
    "evaluation_period": 5000,  # steps
    "evaluation_episodes": 10,
    "num_envs": 8,
    "actor_lr": 3e-4,
    "critic_lr": 1e-3,
    "alpha_lr": 1e-3,
    "init_alpha": 0.2,
    "target_entropy": None,  # Will be set to -dim(A) if None
    "actor_update_rate": 2,
    "auto_entropy_tuning": True,
}
