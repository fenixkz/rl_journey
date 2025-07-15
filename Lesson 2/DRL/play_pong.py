import gymnasium as gym
import numpy as np
# Import RecordVideo wrapper
from gymnasium.wrappers import RecordVideo
import time
import random
import os # For creating directories
import ale_py
import torch
import torch.nn as nn
from D3QN import D3QN





ENV_NAME = "ALE/Pong-v5"
VIDEO_FOLDER = "videos/pong/" # Folder to save videos
RECORD_ALL_EPISODES = True # Set to False to record only cubic episodes

os.makedirs(VIDEO_FOLDER, exist_ok=True)

# --- Environment Setup ---
print(f"Creating environment: {ENV_NAME}")
# Base environment MUST support rgb_array rendering for the wrapper
env = gym.make(ENV_NAME, render_mode="rgb_array")

# --- Apply RecordVideo Wrapper ---
# Define the trigger function
if RECORD_ALL_EPISODES:
    trigger = lambda episode_id: True # Record every episode
else:
    trigger = None # Use default cubic trigger

env = RecordVideo(
    env,
    video_folder=VIDEO_FOLDER,
    episode_trigger=trigger,
    name_prefix=f"breakout-episode" # Optional: customize filename
)
print(f"Applied RecordVideo wrapper, saving to: {VIDEO_FOLDER}")


env = gym.wrappers.AtariPreprocessing(
    env,
    noop_max=0,
    frame_skip=1,
    terminal_on_life_loss=True, # End episode on life loss
    screen_size=84,
    grayscale_obs=True # Agent still needs grayscale
)
print("Applied AtariPreprocessing wrapper.")

env = gym.wrappers.FrameStackObservation(env, stack_size=4) # Agent needs stack
print("Applied FrameStackObservation wrapper.")

agent = D3QN(
    env = env,
    name = "Pong",
    model_type = 'CNN',
)

# --- Interaction Loop (Example with Random Agent) ---
num_episodes = 5 # Record 5 episodes
max_steps_per_episode = 500

# --- Visualization Loop ---
num_episodes_to_watch = 3
print(f"\nStarting visualization for {num_episodes_to_watch} episodes...")

try:
    for episode in range(num_episodes_to_watch):
        episode_reward = 0
        print(f"\n--- Episode {episode + 1} ---")
        agent.validate(n = 1, render=True) # Validate the agent
        print(f"Episode {episode + 1} finished . Reward: {episode_reward}")

except KeyboardInterrupt:
    print("\nVisualization interrupted by user.")
except Exception as e:
    print(f"\nAn error occurred during visualization: {e}")
    import traceback
    traceback.print_exc()
finally:
    # --- IMPORTANT: Close the environment ---
    print("Closing environment.")
    env.close()
