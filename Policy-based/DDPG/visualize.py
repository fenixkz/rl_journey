import gymnasium as gym
import torch
from model import DDPGAgent
import time

name = "BipedalWalker-v3"

env = gym.make(name, render_mode="human")

agent = DDPGAgent(obs_space=env.observation_space, 
                  action_space=env.action_space)

agent.load_models(name)

total_reward = 0
done = False

state, _ = env.reset()
while not done:
    action = agent.get_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    state = next_state
    env.render()
    time.sleep(0.01)

print(f"Total reward: {total_reward}")