from D3QN import D3QN
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

agent = D3QN(
    env = env,
    name = "CartPole",
    n_step_return=3,
    learning_rate=3e-4,
    gamma = 0.99,
    epsilon = 0.99,
    epsilon_decay=0.999,
    epsilon_min = 0.01,
    batch_size = 32,
    buffer_size = int(1e6),
    target_update_freq = 1000,
    validation_period = 20,
    model_type = 'FC',
    beta_start = 0.4,
    beta_end = 1.0,
    beta_annealing_steps=10000,
    alpha=0.5
)

train_rewards, test_rewards = agent.train(n_episodes=500)

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

