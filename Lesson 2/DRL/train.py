import os
import torch
import gymnasium as gym
from DQN.agent import DQN
from DDQN.agent import DDQN
from DuelingDQN.agent import D3QN
import matplotlib.pyplot as plt
import ale_py
from gymnasium.wrappers import AtariPreprocessing
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.plot_utils import get_figure
from utils.atari_config import ATARI_CONFIGS, DEFAULT_ALGO_PARAMS
from utils.classic_config import CLASSIC_ENV_CONFIG
import argparse
from rich import print as pprint

def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL agent on an Atari environment.")
    parser.add_argument('--env', type=str, default="Pong", help="Name of the Atari environment to train on.")
    parser.add_argument('--agent', type=str, default="DQN", choices=["DQN", "DDQN", "D3QN"], help="Agent to use for training.")
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
    
    env_config = get_env_config(args.env)
    agent_name = args.agent
    env_id = env_config["env_id"]
    env_name = env_id.split("/")[-1]

    # Classify whether or not the env is an Atari game
    is_atari = 'ALE/' in env_id

    if is_atari: # We need extra wrappers for Atari envs
        # Create the environment, explicitly set frame_skip = 1 for Atari games
        env = gym.make(env_id, frameskip = 1)
        env = AtariPreprocessing(
                                env,
                                noop_max=30,        # To make each episode a bit different
                                frame_skip=4,       # The agent makes a move for next 4 frames, i.e. moves left for 4 frames to speed up training
                                screen_size=84,     # Rescale original image to 84x84
                                grayscale_obs=True, # RGB to GrayScale
                                scale_obs=True,     # Normalize the pixel values from [0-255] to [0-1]
                                terminal_on_life_loss=True, # Restart the episode on the first life loss, instead of waiting for all lifes losses
                            )
        env = gym.wrappers.FrameStackObservation(env, stack_size=4) # Stack last 4 frames as the observation to encode the velocity information

        # For any missing algorithm param, add the default value
        for key, default_value in DEFAULT_ALGO_PARAMS.items():
            if key not in env_config:
                env_config[key] = default_value

        # Atari specific hyperparams . Taken from the Nature paper of Mnih
        total_timesteps = int(5e6)                                          # atari needs millions of steps. 5 million is a good target.
        learning_starts = 50000                                             # fill the buffer with 50k random steps before learning.
        buffer_size = int(1e5)                                              # suggested 1M, but because of RAM constaints I use 1e5
        batch_size = 32                                                     # the standard batch size from the nature paper.
        lr = env_config.get("lr", 2.5e-4)                                   # a lower learning rate is crucial for stability.
        target_update_freq = env_config.get("target_update_freq", 10000)    # update the target network every 10,000 learning training steps not global.
        learning_freq = env_config.get("learning_freq", 4)                  # perform one learning update every 4 environment steps.
        
        # epsilon decay over the first 1 million steps 
        min_epsilon = 0.1
        start_epsilon = 1.0
        epsilon_decay_steps = env_config.get("epsilon_decay_steps", int(1e6))  
        eps_decay = (min_epsilon / start_epsilon) ** (1 / epsilon_decay_steps)
        
    else: 
        # For Box2D envs, frameskip arg does not exist, so we have to separate 
        env = gym.make(env_id)
        # Classic examples specific hyperparams
        total_timesteps = int(3e5)                           # these examples need less time for training
        learning_starts = 0                                  # we can start learning immediately
        buffer_size = int(1e5)                               # less memory required
        batch_size = 64                                      # a good balance 
        lr = 1e-3                                            # a higher lr is acceptable
        target_update_freq = 2000                            # update more frequently
        learning_freq = 1                                    # learn every step
        min_epsilon = 0.1                                    # at least 10% of actions are random 
        start_epsilon = 1.0                                  # start with full exploration
        epsilon_decay_steps = int(total_timesteps * 0.6)     # decay to min epsilon in 60% of total steps
        eps_decay = (min_epsilon / start_epsilon) ** (1 / epsilon_decay_steps)
    # To record statistics of each episode in info 
    env = gym.wrappers.RecordEpisodeStatistics(env) 
    
    # Common hyperparams
    device = "cuda" if torch.cuda.is_available() else "cpu"   # which device to use for train
    mean_n = 50                                               # mean of rewards over these many episodes

    if agent_name == "DQN":
        pprint(f"[bold yellow] Using DQN agent")
        agent = DQN(
            env=env,
            is_atari=is_atari,
            hidden_space=env_config.get('hidden_dim', 512),
            gamma=env_config.get('gamma', 0.99),
            epsilon=start_epsilon,
            epsilon_decay=eps_decay,
            min_epsilon=min_epsilon,
            device=device,
            buffer_size=buffer_size,
            batch_size=batch_size,
            seed=24,
            lr=lr,
            target_update_freq=target_update_freq,
            learning_freq=learning_freq,
            learning_starts = learning_starts,
            solved_reward = env_config.get("solved_reward", 10000)
            )
    elif agent_name == "DDQN":
        pprint(f"[bold yellow] Using Double-DQN agent")
        agent = DDQN(
            env=env,
            is_atari=is_atari,
            hidden_space=env_config.get('hidden_dim', 512),
            gamma=env_config.get('gamma', 0.99),
            epsilon=start_epsilon,
            epsilon_decay=eps_decay,
            min_epsilon=min_epsilon,
            device=device,
            buffer_size=buffer_size,
            batch_size=batch_size,
            seed=24,
            lr=lr,
            target_update_freq=target_update_freq,
            learning_freq=learning_freq,
            learning_starts = learning_starts,
            solved_reward = env_config.get("solved_reward", 10000)
            )
    elif agent_name == "D3QN":
        pprint(f"[bold yellow] Using Dueling-Double-DQN agent")
        agent = D3QN(
            env=env,
            is_atari=is_atari,
            hidden_space=env_config.get('hidden_dim', 512),
            gamma=env_config.get('gamma', 0.99),
            epsilon=start_epsilon,
            epsilon_decay=eps_decay,
            min_epsilon=min_epsilon,
            device=device,
            buffer_size=buffer_size,
            batch_size=batch_size,
            seed=24,
            lr=lr,
            target_update_freq=target_update_freq,
            learning_freq=learning_freq,
            learning_starts = learning_starts,
            solved_reward = env_config.get("solved_reward", 10000)
            )
    else:
        raise ValueError("Unsupport agent type")
    
    
    mean_rewards = []
    std_rewards = []

    save_path = f"{agent_name}/results/{env_name}"
    os.makedirs(save_path, exist_ok=True)

    def save_progress_to_file():
        """A non-interactive function to save model and plot to file."""
        print("\nSaving model and plotting results to file...")
        agent.save_model(save_path)
        if mean_rewards and std_rewards:
            fig = get_figure(mean_rewards, std_rewards, num_episodes=mean_n)
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

    training_completed_successfully = False
    try:
        agent.train(mean_rewards, std_rewards, max_steps=total_timesteps, mean_n_episodes=mean_n)
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
        final_fig = get_figure(mean_rewards, std_rewards, num_episodes=mean_n)
        plt.show()

if __name__=='__main__':
    args = parse_args()
    main(args)