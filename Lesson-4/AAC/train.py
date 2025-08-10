import multiprocessing as mp
import gymnasium as gym
import numpy as np
import sys
import argparse
from core.agent import AACAgent
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from rich import print as pprint
from rich.console import Console
from config.AAC_CONFIG import AAC_CONFIG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.plot_utils import get_figure
from utils.classic_config import CLASSIC_ENV_CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description="Train a AAC agent.")
    parser.add_argument('--env', type=str, default="cartpole", help="Name of the environment to train on.")
    parser.add_argument('--seed', type=int, default = 224)

    
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
    
    if env_name.lower() in CLASSIC_ENV_CONFIG:
        return CLASSIC_ENV_CONFIG[env_name.lower()]
    
    raise ValueError(f"Unknown environment: '{env_name}'. "
                    f"Available environments: {list(CLASSIC_ENV_CONFIG.keys())}")

def main(args):

    # --- Parse args and parameters, create env and agent ---
    # Parse args
    env_name = args.env
    seed = args.seed

    # Get the config for that specific env 
    env_config = get_env_config(env_name=env_name)
    env_id = env_config['env_id']
    # Is it ATARI env or not

    # Some hyperparameters
    config = AAC_CONFIG
    max_episodes = config.get("max_episodes", 5000) # Total number of learning steps
    actor_lr = config.get("actor_lr", 3e-4) # Learning rate for actor
    critic_lr = config.get("critic_lr", 1e-3) # Learning rate for critic
    gamma = config.get("gamma", 0.9) # Discount rate
    use_normalization = config.get("use_normalization", False)
    use_entropy = config.get("use_entropy", True)
    entropy_coef = config.get("entropy_coef", 1e-2)

    # Create the gym env
    env = gym.make(env_id) 
    
    # Create the agent 
    agent = AACAgent(env=env,
                    solved_threshold=env_config['solved_reward'],
                    gamma = gamma,
                    hidden_size=env_config.get('hidden_dim', 128),
                    actor_lr = actor_lr,
                    critic_lr = critic_lr,
                    seed = seed,
                    use_normalization = use_normalization,
                    use_entropy = use_entropy,
                    entropy_coef = entropy_coef
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
                     window_size=100)
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
    title = f":rocket: :rocket: :rocket: [bold red] Training {env_id} with Advantage Actor-Critic [/bold red] :rocket: :rocket: :rocket:"
    console.print(title, justify="center")
    
    training_completed_successfully = False
    try:
        agent.train(all_rewards=all_rewards,
                    max_episodes=max_episodes
                )
        training_completed_successfully = True
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
    finally:
        # This block will execute on normal completion, Ctrl+C, or a different error.
        save_progress()

    
    # Evaluate policy: use deterministic actions
    print("\nEvaluating trained policy...")

    # --- Display the Plot (Only on Normal Completion) ---
    if training_completed_successfully:
        pprint("Training completed successfully. Displaying final plot.")
        # Re-create the figure from the final data and show it.
        fig = get_figure(all_rewards=all_rewards, 
                     solved_threshold=env_config['solved_reward'],
                     window_size=100)
        plt.show()

    env.close()

if __name__=='__main__':
    args = parse_args()
    main(args)