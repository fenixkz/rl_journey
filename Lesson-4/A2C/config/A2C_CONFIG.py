A2C_CONFIG = {
    "num_envs": 16,
    "gamma": 0.99,
    "lr": 5e-4,
    "critic_scale": 0.5,
    "hidden_dim": 512,
    "use_normalization": True,
    "use_entropy": True,
    "entropy_coef": 1e-2,
    "max_steps": 1e7,
    "num_steps": 128,
    "gae_lambda": 0.95,
    "tau": 0.05,  # Set it to 1 to use only one critic
}
