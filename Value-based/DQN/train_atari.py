from D3QN import D3QN
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import ale_py
from gymnasium.wrappers import AtariPreprocessing


ENV_NAME = "ALE/Pong-v5"
# ENV_NAME = "ALE/Breakout-v5"
name = ENV_NAME.split('/')[-1]

env = gym.make(ENV_NAME)
env = AtariPreprocessing(
    env,
    noop_max=0,
    frame_skip=1, 
    screen_size=84,
    grayscale_obs=True,
    scale_obs=False,
    terminal_on_life_loss=True,
)
env = gym.wrappers.FrameStackObservation(env, stack_size=4)

print(f"Observation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space}")

agent = D3QN(
    env = env,
    name = name,
    n_step_return=3,
    learning_rate=3e-4,
    gamma = 0.99,
    epsilon = 0.99,
    epsilon_decay=0.9999,
    epsilon_min = 0.01,
    batch_size = 32,
    buffer_size = int(1e6),
    target_update_freq = 1000,
    validation_period = 20,
    model_type = 'CNN',
    beta_start = 0.4,
    beta_end = 1.0,
    beta_annealing_steps=10000,
    alpha=0.5
)

train_rewards, test_rewards = agent.train(n_episodes=100000)

# --- Plotting ---
window_size = 50 # Use a larger window for smoother average
# Define moving average function if not defined elsewhere
def moving_avg(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if len(test_rewards) >= window_size:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(window_size - 1, len(test_rewards)), moving_avg(test_rewards, window_size), label=f"{window_size}-ep moving avg reward")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Episode Rewards (Moving Average)")
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    plt.show()
else:
    print("Not enough episodes to plot moving average.")

env.close()


