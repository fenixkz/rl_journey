import torch
import multiprocessing as mp
import gymnasium as gym
import numpy as np
import time
import sys
import argparse
from core.agent import ESAgent
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from rich import print as pprint
from rich.console import Console
from config.ES_CONFIG import ES_CONFIG, ES_ATARI_CONFIG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.plot_utils import get_figure
from utils.classic_config import CLASSIC_ENV_CONFIG
from utils.atari_config import ATARI_CONFIGS
from utils.prepare_atari_env import get_atari_env

def parse_args():
    parser = argparse.ArgumentParser(description="Train an ES agent.")
    parser.add_argument('--env', type=str, default="cartpole", help="Name of the environment to train on.")
    parser.add_argument('--seed', type=int, default=224)
    args = parser.parse_args()
    return args

def get_env_config(env_name: str) -> dict:
    """
    Retrieves the configuration for a given environment name.
    
    Args:
        env_name: The short name of the environment (e.g., "cartpole").
        
    Returns:
        A dictionary containing the environment's configuration.
        
    Raises:
        ValueError: If the short name is not found in the configs.
    """
    if env_name.lower() in ATARI_CONFIGS:
        return ATARI_CONFIGS[env_name.lower()]
    
    if env_name.lower() in CLASSIC_ENV_CONFIG:
        return CLASSIC_ENV_CONFIG[env_name.lower()]
    
    raise ValueError(f"Unknown environment: '{env_name}'. "
                    f"Available environments: {list(ATARI_CONFIGS.keys()) + list(CLASSIC_ENV_CONFIG.keys())}")

def main(args):

    # --- Parse args and parameters, create env and agent ---
    # Parse args
    env_name = args.env
    seed = args.seed

    # Get the config for that specific env 
    env_config = get_env_config(env_name=env_name)
    env_id = env_config['env_id']
    # Is it ATARI env or not
    is_atari = 'ALE/' in env_id
    
    # ES hyperparameters
    config = ES_ATARI_CONFIG if is_atari else ES_CONFIG
    training_generations = config.get("training_generations", 100)  # Number of evolution steps
    population_size = config.get("population_size", 50)             # N: Number of "mutants" in a generation
    noise_std = config.get("noise_std", 0.01)                       # sigma: The "mutation" strength
    learning_rate = config.get("learning_rate", 0.01)               # alpha: How fast the parent evolves
    
    # Create the gym env
    env = gym.make(env_id) if not is_atari else get_atari_env(env_id)
    
    # Create the agent
    agent = ESAgent(env=env,
                    solved_threshold=env_config['solved_reward'],
                    is_atari=is_atari,
                    hidden_dim=env_config.get('hidden_dim', 256),
                    seed=seed
                    )
   
    # Initialize a list of all reward for plotting it later
    all_rewards = []

    # Path where to save the progress
    save_path = f"results/{env_id}"
    os.makedirs(save_path, exist_ok=True)
    
    # Function to save the progress
    def save_progress():
        """A non-interactive function to save model and plot to file."""
        pprint("\nSaving model and plotting results to file...")
        agent.save_policy(save_path)
        if all_rewards:
            fig = get_figure(all_rewards=all_rewards, 
                     solved_threshold=env_config['solved_reward'],
                     window_size=population_size)
            pprint("Generated the figure")
            save_file_path = os.path.join(save_path, "rewards.jpg")
            try:
                fig.savefig(save_file_path)
                pprint(f"Plot saved to {save_file_path}")
            except Exception as e:
                pprint(f"Could not save plot: {e}")
            finally:
                plt.close(fig)

    console = Console()
    title = f":rocket: :rocket: :rocket: [bold red] Training {env_id} with ES [/bold red] :rocket: :rocket: :rocket:"
    console.print(title, justify="center")
    
    training_completed_successfully = False
    try:
        agent.train(all_rewards=all_rewards,
                num_epochs=training_generations,
                population_size=population_size,
                noise_std=noise_std,
                learning_rate=learning_rate
                )
        training_completed_successfully = True
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
    finally:
        # This block will execute on normal completion, Ctrl+C, or a different error.
        save_progress()


    # --- Display the Plot (Only on Normal Completion) ---
    if training_completed_successfully:
        pprint("Training completed successfully. Displaying final plot.")
        # Re-create the figure from the final data and show it.
        fig = get_figure(all_rewards=all_rewards, 
                     solved_threshold=env_config['solved_reward'],
                     window_size=population_size)
        plt.show()

    env.close()

if __name__=='__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass # The start method can only be set once.
    args = parse_args()
    main(args)
