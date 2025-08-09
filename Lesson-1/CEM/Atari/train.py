import torch
import multiprocessing as mp
import gymnasium as gym
import numpy as np
import time
import sys
import argparse
from agent import EnhancedCEMAgent
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from rich import print as pprint
from rich.console import Console
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from config.CEM_CONFIG import CEM_CONFIG, CEM_ATARI_CONFIG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.plot_utils import get_figure
from utils.classic_config import CLASSIC_ENV_CONFIG
from utils.atari_config import ATARI_CONFIGS
from utils.atari_utils import get_atari_env

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CEM agent.")
    parser.add_argument('--env', type=str, default="cartpole", help="Name of the environment to train on.")
    parser.add_argument('--seed', type=int, default = 224)
    parser.add_argument('--cpu-usage', type=float, default = 0.5, help="Part of total CPU cores to use for training for multi-processing, 1 - all cores, 0.1 - 10 percent. Only used for Atari envs.")
    
    # CEMAtari exploration parameters (optional)
    parser.add_argument('--no-warm-start', action='store_true', help="Disable warm-start exploration for Atari")
    parser.add_argument('--warm-start-episodes', type=int, default=30, help="Number of warm-start episodes for Atari")
    parser.add_argument('--warm-start-epsilon', type=float, default=0.3, help="Epsilon for warm-start exploration")
    parser.add_argument('--no-adaptive-percentile', action='store_true', help="Disable adaptive percentile for Atari")
    parser.add_argument('--min-percentile', type=int, default=30, help="Minimum percentile when using adaptive adjustment")
    parser.add_argument('--no-noise-injection', action='store_true', help="Disable noise injection for Atari")
    parser.add_argument('--noise-std', type=float, default=0.1, help="Standard deviation for noise injection")
    
    args = parser.parse_args()
    return args

def get_env_config(env_name: str) -> dict:
    """
    Retrieves the configuration for a given environment  name.
    
    Args:
        env_name: The short name of the environment (e.g., "pong").
        
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
    cpu_usage = max(args.cpu_usage, 0.1)

    # Get the config for that specific env 
    env_config = get_env_config(env_name=env_name)
    env_id = env_config['env_id']
    # Is it ATARI env or not
    is_atari = 'ALE/' in env_id

    # Some hyperparameters
    config = CEM_ATARI_CONFIG if is_atari else CEM_CONFIG
    training_epochs = config.get("training_epochs", 100) # Total number of learning steps
    num_episodes = config.get("num_episodes", 50) # Number of episodes to play to apply CEM filtering
    percentile = config.get("percentile", 60) # Percentile of elite samples to filter
    learning_rate = config.get("lr", 1e-3) # Learning rate
    # Create the gym env
    env = gym.make(env_id) if not is_atari else get_atari_env(env_id)
    
    # Create the agent - use CEMAtari for Atari environments
    pprint(f"[bold cyan]Using EnhancedCEMAgent with exploration enhancements for {env_id}[/bold cyan]")
    agent = EnhancedCEMAgent(env=env,
                    solved_threshold=env_config['solved_reward'],
                    is_atari=is_atari,
                    hidden_dim=env_config.get('hidden_dim', 512),
                    lr = learning_rate,
                    seed = seed,
                    cpu_usage = cpu_usage,
                    # Atari-specific exploration parameters (from command line args)
                    use_warm_start=not args.no_warm_start,
                    warm_start_episodes=args.warm_start_episodes,
                    warm_start_epsilon=args.warm_start_epsilon,
                    use_adaptive_percentile=not args.no_adaptive_percentile,
                    min_percentile=args.min_percentile,
                    use_noise_injection=not args.no_noise_injection,
                    noise_std=args.noise_std
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
                     window_size=num_episodes)
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
    title = f":rocket: :rocket: :rocket: [bold red] Training {env_id} with CEM [/bold red] :rocket: :rocket: :rocket:"
    console.print(title, justify="center")
    
    training_completed_successfully = False
    try:
        agent.train(all_rewards=all_rewards,
                num_epochs=training_epochs,
                num_episodes=num_episodes,
                percentile=percentile
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
                     window_size=num_episodes)
        plt.show()

    env.close()

if __name__=='__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass # The start method can only be set once.
    args = parse_args()
    main(args)