import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from D3QN import D3QNCNN
import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import FireResetEnv
import ale_py
import time # Import time for sleep
import random # Import random for epsilon-greedy if needed (though epsilon=0 here)

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

gym.register_envs(ale_py)
# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
ENV_NAME = "ALE/Pong-v5"
MODEL_PATH = "best_model_d3qn.pth" # Make sure this path is correct

# --- Environment Setup ---
print(f"Creating environment: {ENV_NAME} with render_mode='human'")
env = gym.make(ENV_NAME, render_mode='human') # Use human render mode
# Apply the same wrappers as training
env = FireResetEnv(env)
env = gym.wrappers.AtariPreprocessing(
    env,
    noop_max=0,
    frame_skip=1,
    terminal_on_life_loss=True, # Usually True for evaluation
    screen_size=84,
    grayscale_obs=True # Agent expects grayscale
)
env = gym.wrappers.FrameStackObservation(env, stack_size=4) # Agent expects stack
print("Environment created and wrapped.")
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
