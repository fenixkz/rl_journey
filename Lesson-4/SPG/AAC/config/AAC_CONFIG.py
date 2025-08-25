AAC_CONFIG = {
    "gamma": 0.99,
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "hidden_dim": 64,
    "use_normalization": True,
    "use_entropy": True,
    "entropy_coef": 1e-2,
    "max_steps": 1e6,
    "num_steps": 10,
    "gae_lambda": 0.95,
    "tau": 0.05,  # Set it to 1 to use only one critic
}
