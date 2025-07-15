import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.plot_utils import get_figure
import gymnasium as gym
from agent import DQN
import matplotlib.pyplot as plt

# --- Setup ---
env_id = "LunarLander-v3"
env = gym.make(env_id)
env = gym.wrappers.RecordEpisodeStatistics(env) 
device = "cuda" if torch.cuda.is_available() else "cpu"

agent = DQN(
    env=env,
    hidden_space=64,
    gamma=0.99,
    epsilon=1,
    epsilon_decay=0.999,
    min_epsilon=0.05,
    device=device,
    buffer_size=1e6,
    batch_size=256,
    seed=24,
    lr=3e-4,
    target_update_freq=500,
    learning_freq=1,
    learning_starts=0
    )

mean_n_episodes = 50

mean_rewards = []
std_rewards = []

save_path = f"results/{env_id}"
os.makedirs(save_path, exist_ok=True)

def save_progress_to_file():
    """A non-interactive function to save model and plot to file."""
    print("\nSaving model and plotting results to file...")
    agent.save_model(env_id)
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
    agent.train(mean_rewards, std_rewards, max_steps=100000, mean_n_episodes=mean_n_episodes)
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

