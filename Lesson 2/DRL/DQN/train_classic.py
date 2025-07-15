import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.plot_utils import get_figure
import gymnasium as gym
from agent import DQN
import matplotlib.pyplot as plt

# Hyperparams
# --------
MAX_STEPS = int(1e5) # Total number of steps to play
MIN_EPSILON = 0.05 # 5% chance of random action
START_EPSILON = 1.0 # 100% chance of random action
EPSILON_DECAY_FRACTION = 0.60 # The fraction of total episodes over which to decay epsilon
EPS_DECAY = (MIN_EPSILON / START_EPSILON) ** (1 / int(MAX_STEPS * EPSILON_DECAY_FRACTION)) # A formula for decay to anneal epsilon from start to min epsilon in fraction of max episodes
MEAN_N = 50 # Mean of rewards over these many episodes

# --- Setup ---
env_id = "LunarLander-v3"
env = gym.make(env_id)
env = gym.wrappers.RecordEpisodeStatistics(env) 
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
    target_update_freq=500,
    learning_freq=1,
    learning_starts=0
    )


mean_rewards = []
std_rewards = []

save_path = f"results/{env_id}"
os.makedirs(save_path, exist_ok=True)

def save_progress_to_file():
    """A non-interactive function to save model and plot to file."""
    print("\nSaving model and plotting results to file...")
    agent.save_model(env_id)
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
    agent.train(mean_rewards, std_rewards, max_steps=100000, mean_n_episodes=MEAN_N)
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

