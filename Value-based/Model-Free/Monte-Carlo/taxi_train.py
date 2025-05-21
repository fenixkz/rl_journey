import numpy as np
import gymnasium as gym
from monte_carlo import MCLearning
import matplotlib.pyplot as plt
env = gym.make("Taxi-v3", render_mode="rgb_array")

observation_space = env.observation_space # Not directly used in your MC for Q-table keys
action_space = range(env.action_space.n)

gamma = 0.9
initial_epsilon = 1.0 # Start with full exploration
min_epsilon = 0.01
# Epsilon decay factor - try 0.995 for 1000 episodes, or 0.999 for more episodes
epsilon_decay_factor = 0.9995 # Adjusted for potentially better exploration over 1000 episodes
alpha = 0.1 # Learning rate

num_episodes = 10000 

agent_mc = MCLearning(alpha, gamma, initial_epsilon, action_space)
rewards_mc = []

for e in range(num_episodes):
    state_history = []
    # state_action_count = {} # Not used in this version of MC update, but fine to keep if for other analysis
    state, info = env.reset() # Use both state and info as per new Gymnasium API
    done = False
    total_reward_episode = 0 # Renamed for clarity

    # Play one episode til termination and gather data
    while not done:
        action = agent_mc.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state_history.append((state, action, reward))
        # state_action_count[(state, action)] = state_action_count.get((state, action), 0) + 1 # If you want to use it later
        state = next_state
        total_reward_episode += reward

    if (e + 1) % 100 == 0: # Print every 100 episodes
        print(f"Episode {e+1}, total reward: {total_reward_episode}, epsilon: {agent_mc.epsilon:.4f}")
    rewards_mc.append(total_reward_episode)

    # Decay epsilon
    agent_mc.epsilon = max(min_epsilon, agent_mc.epsilon * epsilon_decay_factor)

    # First-Visit MC Update
    visited_sa_pairs_in_episode = set()
    total_return_g = 0 
    # Calculate the return from the state_history (in reverse)
    for state_hist, action_hist, reward_hist in state_history[::-1]:
        total_return_g = reward_hist + gamma * total_return_g
        if (state_hist, action_hist) not in visited_sa_pairs_in_episode:
            agent_mc.update(state_hist, action_hist, total_return_g)
            visited_sa_pairs_in_episode.add((state_hist, action_hist))

print(f"Average reward of last 10 episodes for MC-Learning: {np.mean(rewards_mc[-10:])}")
window_size = 50 # Increased window size for smoother plot over more episodes
if len(rewards_mc) >= window_size:
    moving_avg = lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size - 1, len(rewards_mc)), moving_avg(rewards_mc), label=f"MC {window_size}-episode moving average")
    plt.xlabel("Episodes")
    plt.ylabel(f"Moving Average Reward ({window_size} episodes)")
    plt.title("Monte Carlo Learning Performance on Taxi-v3")
    plt.legend()
    plt.show()
else:
    print("Not enough episodes to plot moving average with current window size.")

env.close()