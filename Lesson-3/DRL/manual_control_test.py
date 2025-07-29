import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import time
from pynput import keyboard
from gymnasium.wrappers import AtariPreprocessing

# It's good practice to register the ALE environments.
try:
    gym.register_envs(ale_py)
except Exception:
    print("ale_py environments already registered.")

# --- Keyboard Control Setup ---

# Global variable to store the current action from the keyboard
# Default action is 0 (NOOP)
current_action = 0

def on_press(key):
    """Handles key press events."""
    global current_action
    if key == keyboard.Key.left:
        # Action for LEFT in Breakout is 3
        current_action = 3
    elif key == keyboard.Key.right:
        # Action for RIGHT in Breakout is 2
        current_action = 2
    elif key == keyboard.Key.space:
        # Action for FIRE in Breakout is 1
        current_action = 1

def on_release(key):
    """Handles key release events."""
    global current_action
    # When a key is released, default to NOOP action
    current_action = 0

# Start a non-blocking listener for keyboard events
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


# --- Main Visualization Script ---

# 1. Create the environment and apply the same wrappers used for training
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)
env = AtariPreprocessing(
    env,
    noop_max=30,
    frame_skip=4,
    screen_size=84,
    grayscale_obs=True,
    scale_obs=False, # Keep as [0,255] for direct visualization
    terminal_on_life_loss=True,
)
# Note: We don't need RecordEpisodeStatistics for manual play
env = gym.wrappers.FrameStackObservation(env, stack_size=4)

action_meanings = env.unwrapped.get_action_meanings()
print(action_meanings)
exit()
# Reset the environment to get the first observation
observation, info = env.reset(seed=42)

# 2. Set up the Matplotlib plot for interactive display
plt.ion()
fig, ax = plt.subplots()
# The rendered frame will be the original one, but the observation is processed
# To visualize what the agent sees, we plot the observation itself.
# Since it's grayscale and stacked, we'll just show the last frame in the stack.
frame = observation[-1] # Get the last frame from the stack
img = ax.imshow(frame, cmap='gray') # Use a grayscale colormap
plt.axis('off')

print("Starting visualization. Use Left/Right arrow keys to move, Space to fire.")
print("The window now shows the 84x84 grayscale image the agent would see.")
print("Press Ctrl+C in the terminal to stop.")

# Variable to track the score for the current episode
total_reward = 0
episode_count = 1

try:
    while True: # Loop indefinitely until interrupted
        # Use the action from the keyboard listener
        action = 3

        # Step through the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate reward
        total_reward += reward

        # Render the new frame and update the plot data
        frame = observation[-1] # Get the last frame from the new observation stack
        img.set_data(frame)
        
        ax.set_title(f"Episode {episode_count} | Score: {total_reward}")
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        time.sleep(0.5)

        # If the episode has ended, reset and log the score
        if terminated or truncated:
            print(f"Episode {episode_count} finished! Final Score: {total_reward}")
            total_reward = 0 # Reset score for the next episode
            episode_count += 1
            observation, info = env.reset()
            # Auto fire button to help training
            observation, _, _, _, _ = env.step(1)
            print(f"Starting new episode ({episode_count})...")


except KeyboardInterrupt:
    print("\nVisualization stopped by user.")
finally:
    # Clean up
    listener.stop()
    env.close()
    plt.ioff()
    print("Environment closed.")

