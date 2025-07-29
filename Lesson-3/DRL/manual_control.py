import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import time
from pynput import keyboard

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

# 1. Create the environment with render_mode="rgb_array"
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

# Reset the environment to get the first observation
observation, info = env.reset(seed=42)

# 2. Set up the Matplotlib plot for interactive display
plt.ion()
fig, ax = plt.subplots()
frame = env.render()
img = ax.imshow(frame)
plt.axis('off')

print("Starting visualization. Use Left/Right arrow keys to move, Space to fire.")
print("Press Ctrl+C in the terminal to stop.")

# Variable to track the score for the current episode
total_reward = 0

try:
    while True: # Loop indefinitely until interrupted
        # Use the action from the keyboard listener
        action = current_action

        # Step through the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate reward
        total_reward += reward

        # Render the new frame and update the plot data
        frame = env.render()
        img.set_data(frame)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        time.sleep(0.02)

        # If the episode has ended, reset and log the score
        if terminated or truncated:
            print(f"Episode finished! Total Reward: {total_reward}")
            total_reward = 0 # Reset score for the next episode
            observation, info = env.reset()
            print("Starting new episode...")


except KeyboardInterrupt:
    print("\nVisualization stopped by user.")
finally:
    # Clean up
    listener.stop()
    env.close()
    plt.ioff()
    print("Environment closed.")

