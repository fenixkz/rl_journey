import os
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import ale_py
from gymnasium.wrappers import AtariPreprocessing
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.plot_utils import get_figure
from utils.atari_config import ATARI_CONFIGS
from utils.classic_config import CLASSIC_ENV_CONFIG
import argparse
from rich import print as pprint
from core.agent_registry import registry
from core.configs import AgentConfig, DEFAULT_ATARI_PARAMS, DEFAULT_CLASSIC_PARAMS
from core.DQNBase import DQNBase
def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN-based RL agent on a single env for comparison.")
    parser.add_argument('--env', type=str, default="cartpole", help="Name of the environment to train on.")
    parser.add_argument('--seed', type=int, default = 224, help="Seed for reproducibility")
    parser.add_argument('--start-from', type=int, default = 1, help="Specify the number of the agent to start from, to skip first N agents")
    parser.add_argument('--timeout', type=float, default = 20, help="Maximum training time per agent, in minutes")
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
    for agent_number in range(args.start_from, 9):
        env_name = args.env
        env_config = get_env_config(env_name)
        env_id = env_config["env_id"]
        
        env = gym.make(env_id)
        agent_config = AgentConfig.from_dict(DEFAULT_CLASSIC_PARAMS)
        
        # To record statistics of each episode in info 
        env = gym.wrappers.RecordEpisodeStatistics(env) 
        
        # Common hyperparams
        mean_n = 50 # for performance tracking, calculate mean over this many episodes
        agent_config.seed = args.seed # Overwrite seed

        agent: DQNBase = registry.create_agent(agent_number, env, agent_config, env_config, is_atari=False)
        mean_rewards = []
        std_rewards = []

        save_path = os.path.join("results", "arena", env_name)
        os.makedirs(save_path, exist_ok=True)

        def save_progress_to_file():
            """A non-interactive function to save model and plot to file."""
            print("\nSaving results to file...")
            if mean_rewards and std_rewards:
                fig = get_figure(mean_rewards, std_rewards, num_episodes=mean_n)
                print("Generated the figure")
                save_file_path = os.path.join(save_path, f"{agent.get_name()}.jpg")
                try:
                    fig.savefig(save_file_path)
                    print(f"Plot saved to {save_file_path}")
                except Exception as e:
                    print(f"Could not save plot: {e}")
                finally:
                    plt.close(fig)

        pprint(f"----------------------------- :rocket: :rocket: :rocket: [bold red] Training {env_id} [/bold red] :rocket: :rocket: :rocket: -----------------------------")
        pprint(f"[bold yellow] Using {agent.get_name()} Agent with the following config [/bold yellow]: \n {agent_config}")

        try:
            agent.train(mean_rewards, std_rewards, max_steps=agent_config.max_steps, mean_n_episodes=mean_n, timeout=timeout)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user (Ctrl+C).")
        finally:
            # This block will execute on normal completion, Ctrl+C, or a different error.
            save_progress_to_file()


if __name__=='__main__':
    args = parse_args()
    main(args)