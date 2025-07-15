import numpy as np
import gymnasium as gym
import torch
from trainer import PPOTrainer
import os
import matplotlib.pyplot as plt

SEED = 24  # This number is used to ensure reproducibility, use any integer you like



# Environment configurations with specific hyperparameters
ENVIRONMENTS = {
    'CartPole-v1': {
        'num_envs': 4,
        'num_steps': 128,
        'critic_loss_coef': 1,
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'gae_lambda': 0.95,  # Generalized Advantage Estimation parameter
        'entropy_coef': 0,  # Encourage exploration
        'hidden_size': 64,
        'max_steps': 1e6,
        'num_mini_batches': 2, 
        'clip_epsilon': 0.2,
        'update_epochs': 10,
        'solved_threshold': 475.0,  # Average reward over 100 episodes
        'normalize_advantage': True,
        'anneal_lr': True,
        'clip_values': True,
        'description': 'Classic control - balance pole on cart'
    },
    'MountainCar-v0': {
        'num_envs': 4,
        'num_steps': 128,
        'critic_loss_coef': 0.5,
        'actor_lr': 1e-3,
        'critic_lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,  # Generalized Advantage Estimation parameter
        'entropy_coef': 0.01,  # Encourage exploration
        'hidden_size': 64,
        'max_steps': 1e6,
        'num_mini_batches': 5, 
        'clip_epsilon': 0.2,
        'update_epochs': 5,
        'solved_threshold': -110.0,  # Average reward over 100 episodes
        'normalize_advantage': True,
        'anneal_lr': True,
        'clip_values': True,
        'description': 'Get car to top of mountain using momentum'
    },
    'LunarLander-v3': {
        'num_envs': 8,
        'num_steps': 5,
        'critic_loss_coef': 0.5,
        'actor_lr': 1e-3,
        'critic_lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,  # Generalized Advantage Estimation parameter
        'entropy_coef': 0.01,  # Encourage exploration
        'hidden_size': 64,
        'max_steps': 1e6,
        'num_mini_batches': 5, 
        'clip_epsilon': 0.2,
        'update_epochs': 5,
        'solved_threshold': 200.0,  # Average reward over 100 episodes
        'normalize_advantage': True,
        'anneal_lr': True,
        'clip_values': True,
        'description': 'Land spacecraft safely on moon surface'
    },
    'Acrobot-v1': {
        'num_envs': 8,
        'num_steps': 5,
        'critic_loss_coef': 0.5,
        'actor_lr': 1e-3,
        'critic_lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,  # Generalized Advantage Estimation parameter
        'entropy_coef': 0.01,  # Encourage exploration
        'hidden_size': 64,
        'max_steps': 1e6,
        'num_mini_batches': 5, 
        'clip_epsilon': 0.2,
        'update_epochs': 5,
        'solved_threshold': -100.0,  # Average reward over 100 episodes
        'normalize_advantage': True,
        'anneal_lr': True,
        'clip_values': True,
        'description': 'Swing up underactuated pendulum'
    }
}

def plot_results(env_name, rewards):
    # Compute rolling mean and standard deviation (std) for rewards
    rewards = np.array(rewards)
    window = 50
    if len(rewards) < window:
        mean_rewards = np.convolve(rewards, np.ones(len(rewards))/len(rewards), mode='valid')
        std_rewards = np.array([rewards.std()] * len(mean_rewards))
    else:
        mean_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        std_rewards = np.array([rewards[i-window+1:i+1].std() for i in range(window-1, len(rewards))])

    episodes = np.arange(len(mean_rewards))

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, mean_rewards, color='green', label='Mean Reward (Rolling Window=50)')
    plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, color='blue', alpha=0.2, label='Std Dev (Window=50)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title("Training Rewards")
    plt.legend()
    plt.grid(True)
    os.makedirs(f"results/{env_name}/figures", exist_ok=True)
    plt.savefig(f"results/{env_name}/figures/rewards.jpg")
    plt.show()

def main():
    import signal
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for env_name, config in ENVIRONMENTS.items():
        
        print(f"Training {env_name}!")
        os.makedirs(f"results/{env_name}", exist_ok=True)
        trainer = PPOTrainer(
            env_name = env_name,
            num_envs = config['num_envs'],
            num_steps = config['num_steps'],
            critic_loss_coef = config['critic_loss_coef'],
            actor_lr = config['actor_lr'],
            critic_lr = config['critic_lr'],
            gamma = config['gamma'],
            gae_lambda = config['gae_lambda'],
            entropy_coef = config['entropy_coef'],
            hidden_size = config['hidden_size'],
            max_steps = config['max_steps'],
            num_mini_batches = config['num_mini_batches'],
            clip_epsilon = config['clip_epsilon'],
            update_epochs = config['update_epochs'],
            normalize_advantage=config['normalize_advantage'],
            anneal_lr = config['anneal_lr'],
            clip_values = config['clip_values'],
            seed = SEED,
            solved_threshold = config['solved_threshold'],
        )

        def handle_exit(signum, frame):
            trainer.agent.save_models(env_name)
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)

        try:
            rewards = trainer.train()
            plot_results(env_name, rewards)
        finally:
            trainer.agent.save_models(env_name)
            
if __name__ == "__main__":
    main()