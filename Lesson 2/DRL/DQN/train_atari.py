import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.plot_utils import get_figure
import gymnasium as gym
from agent import DQN
import matplotlib.pyplot as plt
import ale_py
from gymnasium.wrappers import AtariPreprocessing

# Hyperparams
# --------
MAX_EPISODES = int(1e6) # Total number of episodes to play
MIN_EPSILON = 0.05 # 5% chance of random action
START_EPSILON = 1.0 # 100% chance of random action
EPSILON_DECAY_FRACTION = 0.60 # The fraction of total episodes over which to decay epsilon
EPS_DECAY = (MIN_EPSILON / START_EPSILON) ** (1 / int(MAX_EPISODES * EPSILON_DECAY_FRACTION)) # A formula for decay to anneal epsilon from start to min epsilon in fraction of max episodes

ENV_NAME = "ALE/Pong-v5"
# ENV_NAME = "ALE/Breakout-v5"
name = ENV_NAME.split('/')[-1]

env = gym.make(ENV_NAME)
env = AtariPreprocessing(
    env,
    noop_max=30, # For more variation in the game
    frame_skip=4, # The agent makes a move for next 4 frames, i.e. moves left for 4 frames to speed up training
    screen_size=84, # Rescale original image to 84x84
    grayscale_obs=True, # RGB to GrayScale
    scale_obs=True, # Normalize the pixel values from [0-255] to [0-1]
    terminal_on_life_loss=True, # Do not wait for all lifes to be wasted, end episode everytime the life is gone
)
env = gym.wrappers.FrameStackObservation(env, stack_size=4)

device = "cuda" if torch.cuda.is_available() else "cpu"

agent = DQN(
    env=env,
    hidden_space=64,
    gamma=0.99,
    epsilon=START_EPSILON,
    epsilon_decay=EPS_DECAY,
    min_epsilon=MIN_EPSILON,
    device=device,
    buffer_size=1e6,
    batch_size=256,
    seed=24,
    lr=3e-4,
    update_period=500,
    )

mean_n_episodes = 50

mean_rewards = []
std_rewards = []

save_path = f"results/{ENV_NAME}"
os.makedirs(save_path, exist_ok=True)

def save_progress_to_file():
    """A non-interactive function to save model and plot to file."""
    print("\nSaving model and plotting results to file...")
    agent.save_model(ENV_NAME)
    if mean_rewards and std_rewards:
        fig = get_figure(mean_rewards, std_rewards, num_episodes=50)
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
    agent.train(mean_rewards, std_rewards, max_episodes=MAX_EPISODES, mean_n_episodes=mean_n_episodes)
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
    final_fig = get_figure(mean_rewards, std_rewards, num_episodes=50)
    plt.show()

