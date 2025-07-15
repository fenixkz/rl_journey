import gymnasium as gym
import numpy as np
# Import wrappers needed
from stable_baselines3.common.atari_wrappers import FireResetEnv
# Import RecordVideo wrapper
from gymnasium.wrappers import RecordVideo
import time
import random
import os # For creating directories
import ale_py
import torch
import torch.nn as nn
from D3QN import D3QNCNN

# --- Helper Function ---
def choose_action(online_model: nn.Module, epsilon: float, state: np.ndarray, action_space_n: int, device: str):
    """Chooses action using epsilon-greedy policy."""
    # Use numpy's random generator for consistency if needed, otherwise random is fine
    if random.random() < epsilon:
        return random.choice(range(action_space_n)) # Use range or env.action_space.sample()
    else:
        with torch.no_grad():
            # Ensure state is float32, add batch dim, move to device
            state_tensor = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
            q_values = online_model(state_tensor)
            # Get action index with max q-value
            action = torch.argmax(q_values, dim=1).squeeze().item() # Use dim=1, squeeze, get Python int
            return action



# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
ENV_NAME = "ALE/Pong-v5"
MODEL_PATH = "best_model_d3qn.pth" # Make sure this path is correct
VIDEO_FOLDER = "videos/pong/" # Folder to save videos
RECORD_ALL_EPISODES = True # Set to False to record only cubic episodes
# Ensure the video folder exists
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

# --- Apply Other Wrappers AFTER RecordVideo ---
# (Order can sometimes matter, but usually fine after RecordVideo)
env = FireResetEnv(env)
print("Applied FireResetEnv wrapper.")

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

print("\nEnvironment Created and Wrapped.")
print(f"Action meanings: {env.unwrapped.get_action_meanings()}") # Access underlying env for meanings
action_dim = env.action_space.n

# --- Load Model ---
print(f"Loading model from: {MODEL_PATH}")
# Ensure observation shape is correctly inferred or passed if needed by D3QNCNN
# For FrameStack, shape is (stack_size, height, width)
obs_shape = env.observation_space.shape
print(f"Observation shape for model: {obs_shape}")
online_model = D3QNCNN(obs_shape, action_dim).to(device)
try:
    online_model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) # Use map_location
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Exiting.")
    exit()
except Exception as e:
    print(f"Error loading model state dict: {e}")
    exit()

# --- Set model to evaluation mode ---
online_model.eval()
print("Model set to evaluation mode.")

# --- Interaction Loop (Example with Random Agent) ---
num_episodes = 5 # Record 5 episodes
max_steps_per_episode = 500

# --- Visualization Loop ---
num_episodes_to_watch = 3
print(f"\nStarting visualization for {num_episodes_to_watch} episodes...")

try:
    for episode in range(num_episodes_to_watch):
        state, info = env.reset() # Reset returns obs and info dict
        terminated = False
        truncated = False
        done = False
        episode_reward = 0
        step_count = 0
        print(f"\n--- Episode {episode + 1} ---")

        while not done:
            # Render the current state *before* taking the action
            env.render()

            # Add a small delay to make it watchable
            time.sleep(0.03) # Adjust sleep time as needed (e.g., 0.02 to 0.05)

            # Choose action deterministically (epsilon = 0)
            action = choose_action(online_model, 0.0, state, action_dim, device)

            # Step the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated # Check if episode ended

            # Move to next state
            state = next_state
            episode_reward += reward
            step_count += 1

        print(f"Episode {episode + 1} finished after {step_count} steps. Reward: {episode_reward}")

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
