import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import time

# It's good practice to register the ALE environments, though not always strictly necessary
# depending on your gymnasium version.
try:
    gym.register_envs(ale_py)
except Exception:
    print("ale_py environments already registered.")


# --- Main Visualization Script ---

# 1. Create the environment with render_mode="rgb_array"
# This tells the environment to render frames to a NumPy array instead of a window.
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

# Reset the environment to get the first observation
observation, info = env.reset(seed=42)

# 2. Set up the Matplotlib plot for interactive display
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
# Render the first frame and display it in the plot
frame = env.render()
img = ax.imshow(frame)
plt.axis('off') # Hide the axes for a cleaner look

print("Starting visualization. Press Ctrl+C in the terminal to stop.")

try:
    for _ in range(5000): # Loop for a large number of steps
        # --- This is where you would insert your trained agent's policy ---
        # For this example, we'll just take random actions.
        # action = agent.choose_action(observation, deterministic=True)
        action = env.action_space.sample()

        # Step through the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # 3. Render the new frame and update the plot data
        frame = env.render()
        img.set_data(frame)
        
        # Redraw the canvas
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # A small pause is needed to allow the GUI to update and to control the speed
        time.sleep(0.02)

        # If the episode has ended, reset to start a new one
        if terminated or truncated:
            print("Episode finished. Resetting.")
            observation, info = env.reset()

except KeyboardInterrupt:
    print("Visualization stopped by user.")
finally:
    # 4. Clean up
    env.close()
    plt.ioff() # Turn off interactive mode
    print("Environment closed.")

