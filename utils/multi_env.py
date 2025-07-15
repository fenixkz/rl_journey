ENVIRONMENTS = {
    'CartPole-v1': {
        'max_episodes': 2000,
        'solved_threshold': 195.0,  # Average reward over 100 episodes
        'use_entropy': True, # Whether or not use entropy for exploration
        'use_normalization': False, # Whether or not normalize the returns 
        'description': 'Classic control - balance pole on cart'
    },
    'MountainCar-v0': {
        'max_episodes': 3000,  # Harder environment, needs more episodes
        'solved_threshold': -110.0,  # MountainCar has negative rewards
        'use_entropy': True, # Whether or not use entropy for exploration
        'use_normalization': True, # Whether or not normalize the returns 
        'description': 'Get car to top of mountain using momentum'
    },
    'LunarLander-v3': {
        'max_episodes': 2000,
        'solved_threshold': 200.0,
        'use_entropy': True, # Whether or not use entropy for exploration
        'use_normalization': False, # Whether or not normalize the returns 
        'description': 'Land spacecraft safely on moon surface'
    },
    'Acrobot-v1': {
        'max_episodes': 3000,
        'solved_threshold': -100.0,
        'use_entropy': True, # Whether or not use entropy for exploration
        'use_normalization': False, # Whether or not normalize the returns 
        'description': 'Swing up underactuated pendulum'
    }
}

