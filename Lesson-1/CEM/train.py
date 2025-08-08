import multiprocessing as mp
import gymnasium as gym
import numpy as np
import sys
import argparse
from core.agent import CEMAgent
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from rich import print as pprint
from rich.console import Console
from config.CEM_CONFIG import CEM_CONFIG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.plot_utils import get_figure
from utils.classic_config import CLASSIC_ENV_CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CEM agent.")
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
    config = CEM_CONFIG
    training_epochs = config.get("training_epochs", 100) # Total number of learning steps
    num_episodes = config.get("num_episodes", 50) # Number of episodes to play to apply CEM filtering
    percentile = config.get("percentile", 60) # Percentile of elite samples to filter
    learning_rate = config.get("lr", 1e-3) # Learning rate
    # Create the gym env
    env = gym.make(env_id) 
    
    # Create the agent 
    agent = CEMAgent(env=env,
                    solved_threshold=env_config['solved_reward'],
                    hidden_dim=env_config.get('hidden_dim', 128),
                    lr = learning_rate,
                    seed = seed,
                    )

    # --- Visualization: Untrained Policy, only for non-ATARI ---
    
    human_env = gym.make(env_id, render_mode="human") # Create a similar env, but that can visualize the environment using display

    _, total_reward = agent.play_one_episode(human_env, agent, render=True)
    pprint(f"Total reward achieved by untrained policy [bold red]: {total_reward}")
    human_env.close()
   
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

    
    # Evaluate policy: use deterministic actions
    print("\nEvaluating trained policy...")
    
    # --- Visualization: Trained Policy ---
    
    human_env = gym.make(env_id, render_mode="human") # Create a similar env, but that can visualize the environment using display

    _, total_reward = agent.play_one_episode(env=human_env, deterministic=True, render=True)
    pprint(f"Total reward achieved by trained policy [bold green]: {total_reward}")
    human_env.close()
    # --- End of visualization

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
    args = parse_args()
    main(args)