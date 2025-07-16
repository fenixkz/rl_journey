from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import os 
from typing import List
from base.buffer import ReplayBuffer
from base.policy import FCNetwork, CNNNetwork

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DQNAgent(ABC):

    def __init__(self, 
                 env: gym.Env,
                 is_atari: bool = False,
                 hidden_space: int = 128,
                 gamma: float = 0.99,
                 epsilon: float = 1,
                 epsilon_decay: float = 0.9995,
                 min_epsilon: float = 0.05,
                 device: str = "cpu",
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 seed: int = 24,
                 ):
        assert(isinstance(env.action_space, gym.spaces.Discrete)), "Detected non-discrete action space, this class works only with discrete action space problems!"

        # Some hyperparameters
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = device
        self.batch_size = batch_size
        self.is_atari = is_atari

        if not is_atari:
            print(f"Detected a classic (vector) environment, observation space shape: {self.observation_space.shape}, using Fully Connected (FC) Network")
            self.online_model = FCNetwork(self.observation_space.shape[0], self.action_space.n, hidden_space).to(device)
            self.target_model = FCNetwork(self.observation_space.shape[0], self.action_space.n, hidden_space).to(device)
        else:
            print(f"Detected an Atari environment, observation space shape: {self.observation_space.shape}, using Convolutional Neural Network (CNN)")
            self.online_model = CNNNetwork(self.observation_space.shape, self.action_space.n, hidden_space).to(device)
            self.target_model = CNNNetwork(self.observation_space.shape, self.action_space.n, hidden_space).to(device)

        # Initialize a replay buffer 
        self.memory = ReplayBuffer(size=int(buffer_size))
        # Set seed for reproduction of results
        self.seed = seed
        set_seed(seed)
        
    def choose_action(self, state: np.ndarray):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space.n)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_model(state_tensor)
            action = torch.argmax(q_values, dim=1).squeeze().item() 
            return action

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def save_model(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        online_model_path = os.path.join(save_path, "online_model.pth")
        target_model_path = os.path.join(save_path, "target_model.pth")
        torch.save(self.online_model.state_dict(), online_model_path)
        torch.save(self.target_model.state_dict(), target_model_path)
    
    def load_model(self, load_path: str):
        os.makedirs(load_path, exist_ok=True)
        online_model_path = os.path.join(load_path, "online_model.pth")
        target_model_path = os.path.join(load_path, "target_model.pth")
        self.online_model.load_state_dict(torch.load(online_model_path))
        self.target_model.load_state_dict(torch.load(target_model_path))

    def auto_fire(self):
        action_descr = self.env.unwrapped.get_action_meanings()
        fire_action = action_descr.index("FIRE")
        obs, _, _, _, _ = self.env.step(fire_action)
        return obs
    
    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def train(self, mean_rewards: List, std_rewards: List, max_steps: int = 100000, mean_n_episodes: int = 50):
        pass