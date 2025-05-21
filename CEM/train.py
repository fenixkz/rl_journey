import torch
import gymnasium as gym
import numpy as np
import time
from model import Agent
import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from env import MyTrainingEnv
import connect4 as c4
from tqdm import tqdm

def select_elites(states, actions, rewards, percentile):
    reward_threshold = np.percentile(rewards, percentile)
    elite_states = []
    elite_actions = []
    for i in range(len(rewards)):
        if rewards[i] >= reward_threshold:
            elite_states.extend(states[i])
            elite_actions.extend(actions[i])
    return np.array(elite_states), np.array(elite_actions)

# --- Configuration ---
env_id = "LunarLander-v3" # "CartPole-v1" "LunarLander-v0" "ConnectFour-v0" 

device = "cuda" if torch.cuda.is_available() else "cpu"

env = MyTrainingEnv(env_id)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

agent = Agent(obs_dim, act_dim, device).to(device)
n_episodes = 20
training_steps = 100
terminate_after = 5
mean_rewards = []
steps_in_a_row = 0
with tqdm(range(training_steps), desc="Training", postfix={"mean_reward": 0}) as pbar:
    for _ in pbar:
        states, actions, rewards = env.collect_data(n_episodes, agent)
        mean_reward = np.percentile(rewards, 50)
        elite_states, elite_actions = select_elites(states, actions, rewards, percentile=70)
        agent.fit(elite_states, elite_actions)
        mean_rewards.append(mean_reward)
        if mean_reward > 200:
            steps_in_a_row += 1
        else:
            steps_in_a_row = 0
        if steps_in_a_row >= terminate_after:
            print(f"Terminated after {_} steps with mean reward {mean_reward}")
            break
        pbar.set_postfix_str(f"mean_reward: {mean_reward:.2f}")
# Evaluate agent: use argmax instead of sample, play 10 episodes and compute average reward
n_eval = 10
rewards = []
for _ in range(n_eval):
    _, _, reward = env.run_episode(agent, deterministic=True)
    rewards.append(reward)
    print(reward)

print(f"Average reward over {n_eval} episodes: {np.mean(rewards)}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(mean_rewards)
plt.title('Mean Rewards over Training Steps')
plt.xlabel('Training Steps')
plt.ylabel('Mean Reward')
plt.grid(True)
plt.show()

env.close()