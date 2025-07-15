import os
import torch
import gymnasium as gym
from DQN.agent import DQN
from DDQN.agent import DDQN
import matplotlib.pyplot as plt
import ale_py
from gymnasium.wrappers import AtariPreprocessing
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.plot_utils import get_figure
import argparse


parser = argparse.ArgumentParser(description="Train an RL agent on an Atari environment.")
parser.add_argument('--env', type=str, default="Pong", help="Name of the Atari environment to train on.")
parser.add_argument('--agent', type=str, default="DQN", choices=["DQN", "DDQN"], help="Agent to use for training.")
args = parser.parse_args()

if args.env == "Pong":
    ENV_NAME = "ALE/Pong-v5"
elif args.env == "Breakout":
    ENV_NAME = "ALE/Breakout-v5"
elif args.env == "SpaceInvaders":
    ENV_NAME = "ALE/SpaceInvaders-v5"
elif args.env == "Seaquest":
    ENV_NAME = "ALE/Seaquest-v5"
else:
    raise ValueError("Unsupported env name")

ENV_NAME = args.env
AGENT = args.agent

print(f"------------ Training {ENV_NAME} ---------------")
# Hyperparams
# --------
TOTAL_TIMESTEPS = int(10e6)    # Atari needs millions of steps. 10 million is a good target.
LEARNING_STARTS = 50000        # Fill the buffer with 50k random steps before learning.
BUFFER_SIZE = int(1e5)         # A large buffer of 1 million transitions.
BATCH_SIZE = 32                # The standard batch size from the Nature paper.
LR = 1e-4                      # A lower learning rate is crucial for stability.
TARGET_UPDATE_FREQ = 10000     # Update the target network every 10,000 training steps.
LEARNING_FREQ = 4              # Perform one learning update every 4 environment steps.

# Epsilon decay over the first 1 million STEPS (not episodes)
MIN_EPSILON = 0.1
START_EPSILON = 1.0
EPSILON_DECAY_STEPS = 1_000_000
EPS_DECAY = (MIN_EPSILON / START_EPSILON) ** (1 / EPSILON_DECAY_STEPS)
MEAN_N = 50 # Mean of rewards over these many episodes
ENV_NAME = "ALE/Pong-v5" # "ALE/Breakout-v5" ALE/SpaceInvaders-v5

name = ENV_NAME.split('/')[-1]

env = gym.make(ENV_NAME, frameskip=1)
env = AtariPreprocessing(
    env,
    noop_max=30, # For more variation in the game
    frame_skip=4, # The agent makes a move for next 4 frames, i.e. moves left for 4 frames to speed up training
    screen_size=84, # Rescale original image to 84x84
    grayscale_obs=True, # RGB to GrayScale
    scale_obs=True, # Normalize the pixel values from [0-255] to [0-1]
    terminal_on_life_loss=True, # Do not wait for all lifes to be wasted, end episode everytime the life is gone
)
env = gym.wrappers.RecordEpisodeStatistics(env) 
env = gym.wrappers.FrameStackObservation(env, stack_size=4)

device = "cuda" if torch.cuda.is_available() else "cpu"

agent = DQN(
    env=env,
    hidden_space=512,
    gamma=0.99,
    epsilon=START_EPSILON,
    epsilon_decay=EPS_DECAY,
    min_epsilon=MIN_EPSILON,
    device=device,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    seed=24,
    lr=LR,
    target_update_freq=TARGET_UPDATE_FREQ,
    learning_freq=LEARNING_FREQ,
    )

mean_n_episodes = 50

mean_rewards = []
std_rewards = []

save_path = f"{AGENT}/results/{ENV_NAME}"
os.makedirs(save_path, exist_ok=True)

def save_progress_to_file():
    """A non-interactive function to save model and plot to file."""
    print("\nSaving model and plotting results to file...")
    agent.save_model(save_path)
    if mean_rewards and std_rewards:
        fig = get_figure(mean_rewards, std_rewards, num_episodes=MEAN_N)
        print("Generated the figure")
        save_file_path = os.path.join(save_path, "rewards.jpg")
        try:
            fig.savefig(save_file_path)
            print(f"Plot saved to {save_file_path}")
        except Exception as e:
            print(f"Could not save plot: {e}")
        finally:
            plt.close(fig)

training_completed_successfully = False
try:
    agent.train(mean_rewards, std_rewards, max_steps=TOTAL_TIMESTEPS, mean_n_episodes=mean_n_episodes)
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
    final_fig = get_figure(mean_rewards, std_rewards, num_episodes=MEAN_N)
    plt.show()

