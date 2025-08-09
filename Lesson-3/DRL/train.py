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
from utils.atari_utils import get_atari_env
import argparse
from rich import print as pprint
from core.agent_registry import registry
from core.configs import AgentConfig, DEFAULT_ATARI_PARAMS, DEFAULT_CLASSIC_PARAMS
from core.DQNBase import DQNBase
def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL agent on an environment.")
    parser.add_argument('--env', type=str, default="Pong", help="Name of the environment to train on.")
    parser.add_argument('--agent', type=int, default="1", choices=[1, 2, 3, 4, 5, 6, 7, 8], 
                        help="Agent to use for training. \n 1 - DQN \n 2 - Double DQN \n 3 - Double DQN with Prioritized Experience Replay \n 4 - Dueling DQN \
                            \n 5- Dueling DQN with N-step Return and PER \n 6 - Distributional DQN with N-step Return and PER \n \
                            7 - Noisy Net DQN with N-step Return and PER \n 8 - RAINBOW")
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
    if env_name.lower() in ATARI_CONFIGS:
        return ATARI_CONFIGS[env_name.lower()]
    
    if env_name.lower() in CLASSIC_ENV_CONFIG:
        return CLASSIC_ENV_CONFIG[env_name.lower()]
    
    raise ValueError(f"Unknown environment: '{env_name}'. "
                    f"Available environments: {list(ATARI_CONFIGS.keys()) + list(CLASSIC_ENV_CONFIG.keys())}")

def main(args):
    agent_number = args.agent
    env_config = get_env_config(args.env)
    
    env_id = env_config["env_id"]
    env_name = env_id.split("/")[-1]

    # Classify whether or not the env is an Atari game
    is_atari = 'ALE/' in env_id

    if is_atari: # We need extra wrappers for Atari envs
        env = get_atari_env(env_id)

        # Add to the config default params of the algorithm (gamma, batch_size and etc.)
        agent_config = AgentConfig.from_dict(DEFAULT_ATARI_PARAMS)
    else: 
        # For Box2D envs, frameskip arg does not exist, so we have to separate 
        env = gym.make(env_id)
        agent_config = AgentConfig.from_dict(DEFAULT_CLASSIC_PARAMS)
    
    # To record statistics of each episode in info 
    env = gym.wrappers.RecordEpisodeStatistics(env) 
    
    # Common hyperparams
    agent_config.seed = args.seed # Overwrite seed

    agent: DQNBase = registry.create_agent(agent_number, env, agent_config, env_config, is_atari)
    all_rewards = []

    save_path = os.path.join("results", agent.get_name(), env_name)
    os.makedirs(save_path, exist_ok=True)

    def save_progress_to_file():
        """A non-interactive function to save model and plot to file."""
        print("\nSaving model and plotting results to file...")
        agent.save_model(save_path)
        if all_rewards:
            fig = get_figure(all_rewards=all_rewards, 
                             solved_threshold=env_config.get("solved_reward", 100),
                             window_size=50)
            print("Generated the figure")
            save_file_path = os.path.join(save_path, "rewards.jpg")
            try:
                fig.savefig(save_file_path)
                print(f"Plot saved to {save_file_path}")
            except Exception as e:
                print(f"Could not save plot: {e}")
            finally:
                plt.close(fig)

    pprint(f"----------------------------- :rocket: :rocket: :rocket: [bold red] Training {env_id} [/bold red] :rocket: :rocket: :rocket: -----------------------------")
    pprint(f"[bold yellow] Using {agent.get_name()} Agent with the following config [/bold yellow]: \n {agent_config}")

    training_completed_successfully = False
    try:
        agent.train(all_rewards, max_steps=agent_config.max_steps)
        training_completed_successfully = True
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
    finally:
        # This block will execute on normal completion, Ctrl+C, or a different error.
        save_progress_to_file()

    # --- Display the Plot (Only on Normal Completion) ---
    if training_completed_successfully:
        print("\nTraining completed successfully. Displaying final plot.")
        # Re-create the figure from the final data and show it.
        final_fig = get_figure(all_rewards=all_rewards, 
                             solved_threshold=env_config.get("solved_reward", 100),
                             window_size=50)
        plt.show()

if __name__=='__main__':
    args = parse_args()
    main(args)