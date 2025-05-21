import gymnasium as gym
import numpy as np
import time
from CEM.model import Agent
from env import MyTrainingEnv
import torch


# --- Configuration ---
env_id = "CartPole-v1" 

device = "cuda" if torch.cuda.is_available() else "cpu"

env = MyTrainingEnv(env_id)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

agent = Agent(obs_dim, act_dim, device).to(device)


states, actions, rewards = env.collect_data(10, agent)




env.close()