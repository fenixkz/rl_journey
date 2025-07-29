import numpy as np
import gymnasium as gym
from monte_carlo import MCLearning
import matplotlib.pyplot as plt
from collections import deque
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.plot_utils import get_figure
from tqdm import tqdm

# ---------- ENV CONFIG ----------
env_id = "Taxi-v3"
env = gym.make(env_id)
action_space = range(env.action_space.n)

# ---------- HYPERPARAMETERS ----------
gamma = 0.9                            # Discount factor
initial_epsilon = 1.0                  # Start with full exploration
min_epsilon = 0.01                     # 1% chance of picking a random action
epsilon_decay_factor = 0.9995          # Epsilone is decayed by this factor every episode
alpha = 0.05                           # Learning rate / scale of moving average
num_episodes = 10000                   # Total number of episodes to play and train
mean_epsisodes = 50                    # Calculate mean reward for that many episodes, for performance tracking
seed = 224                             # Seed for reproducibility of results

# ---------- AGENT INITIALIZATION ----------
agent_mc = MCLearning(alpha, gamma, initial_epsilon, action_space)

# ---------- METRICS ----------
rewards = deque(maxlen=mean_epsisodes)
mean_rewards = []
std_rewards = []
pbar = tqdm(range(num_episodes), desc="Training", postfix={"mean_reward": "N/A"})

# ---------- TRAIN LOOP ----------
for e in pbar:
    # Initialize an empty list to store the history of (state, action, reward) tuples
    state_history = [] 
    # Reset the environment
    state, info = env.reset(seed=seed) 
    done = False
    total_reward_episode = 0 

    # Play one full episode til termination and gather data
    while not done:
        # Epsilon-greedy strategy for action picking
        action = agent_mc.get_action(state)
        # Apply the action to the env
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # Store the currect tuple in the history
        state_history.append((state, action, reward))
        # Transit
        state = next_state
        total_reward_episode += reward
    
    # Calculate the metrics
    rewards.append(total_reward_episode)
    mean_rewards.append(np.mean(rewards))
    std_rewards.append(np.std(rewards))
    
    # Decay epsilon
    agent_mc.epsilon = max(min_epsilon, agent_mc.epsilon * epsilon_decay_factor)

    postfix = {"mean_reward": f"{np.mean(rewards):.2f}", "eps": f"{agent_mc.epsilon:.3f}"}
    pbar.set_postfix(postfix)

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
pbar.close()