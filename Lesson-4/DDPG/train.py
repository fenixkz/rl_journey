import gymnasium as gym
import numpy as np
from model import DDPGAgent, ReplayBuffer
from collections import deque
import matplotlib.pyplot as plt
import os 


def evaluate_agent(agent, env, num_episodes=10):
    """Runs the agent for a number of episodes with no exploration noise."""
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            # Get action with NO noise
            action = agent.get_action(state, apply_noise=False) 
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)



env_name = "BipedalWalker-v3"
env = gym.make(env_name)
# BipedalWalker-v3
# Pendulum-v1
num_episodes = 5000
initial_noise = 0.1
noise_decay = 0.999
min_noise = 0.01
buffer_size = int(1e6)
batch_size = 512
tau = 5e-3
gamma = 0.99
evaluation_period = 50 # Evaluate every 50 episodes
agent = DDPGAgent(obs_space=env.observation_space, 
                  action_space=env.action_space, 
                  noise_magnitude=initial_noise, 
                  noise_decay=noise_decay, 
                  min_noise=min_noise,
                  gamma = gamma,
                  actor_lr = 5e-4,
                  critic_lr = 1e-3,
                  batch_size=batch_size,
                  tau = tau
                  )
replay_buffer = ReplayBuffer(size=buffer_size)
reward_deque = deque(maxlen=50)
mean_rewards = []
std_rewards = []
eval_rewards = []
for e in range(num_episodes):
    state,_ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action = agent.get_action(state) # action is a np.array
        next_state, reward, terminated, truncated, _ = env.step(action=action)
        done = terminated or truncated
        episode_reward += reward
        replay_buffer.push(state, action, reward, next_state, done)
        if len(replay_buffer) > batch_size:
            agent.learn(replay_buffer.sample(batch_size=batch_size))
        state = next_state
    agent.decay_noise()
    print(f"Episode {e}, Reward: {episode_reward}, Noise: {agent.noise_magnitude}")
    reward_deque.append(episode_reward)
    mean_rewards.append(np.mean(reward_deque))
    std_rewards.append(np.std(reward_deque))
    if (e+1) % evaluation_period == 0: # Time to evaluate
        eval_reward = evaluate_agent(agent, env, num_episodes=10)
        eval_rewards.append(eval_reward)
agent.save_models(env_name)

# Convert your lists to NumPy arrays for easier calculations
mean_rewards = np.array(mean_rewards)
std_rewards = np.array(std_rewards)

# Create an array for the x-axis (episode numbers)
episodes = np.arange(len(mean_rewards))

# Create the plot
plt.figure(figsize=(12, 6)) # Set a good figure size

# Plot the mean reward line
plt.subplot(1,2,1)
plt.plot(episodes, mean_rewards, color='green', label='Mean Training Episodic Reward (Rolling Window=50)')

# Plot the standard deviation area
# y1 is the lower bound, y2 is the upper bound
plt.fill_between(episodes, 
                 mean_rewards - std_rewards, 
                 mean_rewards + std_rewards, 
                 color='green', 
                 alpha=0.2,  # Use alpha for transparency
                 label='Standard Deviation')
plt.xlabel('Episode')
plt.ylabel('Total Train Reward')
plt.title("Training")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(eval_rewards, label="Evaluation rewards")
plt.xlabel('Episode')
plt.ylabel('Total Evaluation Reward')
plt.title("Validation")
plt.grid(True)
# Add labels and a title for clarity
plt.suptitle('DDPG Agent Performance on BipedalWalker-v3')

os.makedirs("figures", exist_ok=True)
plt.savefig(f"figures/{env_name}_rewards.jpg")
plt.show()
