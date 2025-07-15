import numpy as np
import gymnasium as gym
from monte_carlo import MCLearning
import matplotlib.pyplot as plt
from collections import deque
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.plot_utils import get_figure

env_id = "Taxi-v3"
env = gym.make(env_id)

action_space = range(env.action_space.n)

gamma = 0.9
initial_epsilon = 1.0 # Start with full exploration
min_epsilon = 0.01 # 1% chance of picking the random action
epsilon_decay_factor = 0.9995 # We decay by this factor each episode
alpha = 0.1 # Learning rate or scale of moving average

num_episodes = 30000 
mean_epsisodes = 50 # Calculate mean reward for that many episodes
agent_mc = MCLearning(alpha, gamma, initial_epsilon, action_space)
rewards = deque(maxlen=mean_epsisodes)
mean_rewards = []
std_rewards = []
for e in range(num_episodes):
    state_history = []
    state, info = env.reset() 
    done = False
    total_reward_episode = 0 

    # Play one episode til termination and gather data
    while not done:
        action = agent_mc.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state_history.append((state, action, reward))
        state = next_state
        total_reward_episode += reward

    if (e + 1) % 100 == 0: # Print every 100 episodes
        print(f"Episode {e+1}, total reward: {total_reward_episode}, epsilon: {agent_mc.epsilon:.4f}")

    rewards.append(total_reward_episode)
    mean_rewards.append(np.mean(rewards))
    std_rewards.append(np.std(rewards))

    # Decay epsilon
    agent_mc.epsilon = max(min_epsilon, agent_mc.epsilon * epsilon_decay_factor)

    # First-Visit MC Update: Update only once per state, action visit
    visited_sa_pairs_in_episode = set()
    total_return_g = 0 
    # Calculate the return from the state_history (in reverse)
    for state_hist, action_hist, reward_hist in state_history[::-1]:
        total_return_g = reward_hist + gamma * total_return_g
        if (state_hist, action_hist) not in visited_sa_pairs_in_episode:
            agent_mc.learn(state_hist, action_hist, total_return_g)
            visited_sa_pairs_in_episode.add((state_hist, action_hist))

save_path = f"results/{env_id}"
os.makedirs(save_path, exist_ok=True)

fig = get_figure(mean_rewards, std_rewards, num_episodes=mean_epsisodes)
fig.savefig(f"{save_path}/rewards.jpg")
plt.show()
env.close()