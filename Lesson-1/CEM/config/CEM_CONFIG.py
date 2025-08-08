CEM_CONFIG = {
    "training_epochs": 100,
    "num_episodes": 50,
    "percentile": 70,
    "lr": 1e-3,
}

CEM_ATARI_CONFIG = {
    # Training Configuration
    "training_epochs": 800,  # Increased from 100 - Atari needs much longer training
    "num_episodes": 150,     # Increased from 50 - More episodes for stable elite selection
    "percentile": 80,        # Increased from 70 - Higher threshold due to reward variance
    "lr": 3e-4,
}