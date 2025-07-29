CLASSIC_ENV_CONFIG = {
    'cartpole': {
        'env_id': 'CartPole-v1',
        'hidden_dim': 128,
        'solved_reward': 495.0,  # Average reward over 100 episodes
        'use_entropy': True, # Whether or not use entropy for exploration
        'use_normalization': False, # Whether or not normalize the returns 
        'description': 'Classic control - balance pole on cart'
    },
    'mountaincar': {
        'env_id': 'MountainCar-v0',
        'hidden_dim': 64,
        'solved_reward': -110.0,  # MountainCar has negative rewards
        'use_entropy': True, # Whether or not use entropy for exploration
        'use_normalization': True, # Whether or not normalize the returns 
        'description': 'Get car to top of mountain using momentum'
    },
    'lunarlander': {
        'env_id': 'LunarLander-v3',
        'hidden_dim': 128,
        'solved_reward': 200.0,
        'use_entropy': True, # Whether or not use entropy for exploration
        'use_normalization': False, # Whether or not normalize the returns 
        'description': 'Land spacecraft safely on moon surface'
    },
    'acrobot': {
        'env_id': 'Acrobot-v1',
        'hidden_dim': 128,
        'solved_reward': -100.0,
        'use_entropy': True, # Whether or not use entropy for exploration
        'use_normalization': False, # Whether or not normalize the returns 
        'description': 'Swing up underactuated pendulum'
    }
}

