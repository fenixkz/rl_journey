import numpy as np
import gymnasium as gym
from qlearning import QLearningAgent
from sarsa import SARSA
from ev_sarsa import EVSARSA
import matplotlib.pyplot as plt
env = gym.make("Taxi-v3", render_mode = "rgb_array")


observation_space = env.observation_space
action_space = range(env.action_space.n)

gamma = 0.9
epsilon = 0.9
alpha = 0.1
num_episodes = 2000
rewards = {'q-learning': [], 'sarsa': [], 'ev-sarsa': []}

agent = QLearningAgent(alpha = alpha, gamma = gamma, epsilon = epsilon, action_space = action_space)
for e in range(num_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    # Let's decay the epsilon, such that in later epsidodes it used more of greedy policy
    agent.epsilon = max(0.01, agent.epsilon * 0.995)  # Decay epsilon
    rewards['q-learning'].append(total_reward)
    
print(f"Average reward of last 10 episodes for Q-Learning: {np.mean(rewards['q-learning'][-10:])}")

agent = SARSA(alpha = alpha, gamma = gamma, epsilon = epsilon, action_space = action_space)
for e in range(1000):
    state = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_action = agent.get_action(next_state) # Only difference
        agent.update(state, action, reward, next_state, next_action)
        state = next_state
        total_reward += reward
    # print(f"Episode: {e} Reward per episode: {total_reward}")
    # Let's decay the epsilon, such that in later epsidodes it used more of greedy policy
    agent.epsilon *= 0.99
    rewards['sarsa'].append(total_reward)
print(f"Average reward of last 10 episodes for SARSA: {np.mean(rewards['sarsa'][-10:])}")


agent = EVSARSA(alpha = alpha, gamma = gamma, epsilon = epsilon, action_space = action_space)
for e in range(num_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    # Let's decay the epsilon, such that in later epsidodes it used more of greedy policy
    agent.epsilon = max(0.01, agent.epsilon * 0.995)  # Decay epsilon
    rewards['ev-sarsa'].append(total_reward)
print(f"Average reward of last 10 episodes for EV-SARSA: {np.mean(rewards['ev-sarsa'][-10:])}")

import matplotlib.pyplot as plt
import numpy as np

window_size = 10
moving_avg = lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='valid')
plt.figure(figsize=(12, 6))
for key in rewards.keys():
    plt.plot(range(window_size-1, len(rewards[key])), moving_avg(rewards[key]), label=f"{key} {window_size}-episode moving average")
plt.xlabel("Episode")
plt.ylabel(f"Moving Average Reward ({window_size} episodes)")
plt.title("Training rewards per episode")
plt.legend()
plt.show()
env.close()

