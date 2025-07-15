from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import random
import os 
from typing import List


class FCNetwork(nn.Module):
    def __init__(self, obs_space, action_space, hidden_space: int = 128):
        super(FCNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space),
        )
        
    def forward(self, x: torch.Tensor):
        return self.network(x)
    
class CNNNetwork(nn.Module):
    def __init__(self, obs_space, action_space, hidden_space: int = 128):
        super(CNNNetwork, self).__init__()
        C,H,W = obs_space
        self.backbone = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )   
        # Pass a zero tensor to get the final flattened shape of the resulting tensor
        with torch.no_grad():
            feature_size = self.backbone(torch.zeros(1, *obs_space)).view(1, -1).size(1)
        self.regressor = nn.Sequential(
            nn.Linear(feature_size, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space)
        )

    def forward(self, x: torch.Tensor):
        # Extract features from the image
        x = self.backbone(x)
        # Flatten
        x = x.view(x.size(0), -1)
        return self.regressor(x)

class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Replay buffer. Taken from Stable Baselines.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t))
            actions.append(np.array(action))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1))
            dones.append(done)

        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

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
                 hidden_space: int = 128,
                 gamma: float = 0.99,
                 epsilon: float = 1,
                 epsilon_decay: float = 0.9995,
                 min_epsilon: float = 0.05,
                 device: str = "cpu",
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 seed: int = 24
                 ):
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

        # 1. Classical examples, observation_space is a Box and shape is 1D
        if isinstance(self.observation_space, gym.spaces.Box) and len(self.observation_space.shape) == 1:
            print(f"Detected a classic environment, observation space shape: {self.observation_space.shape}, using Fully Connected (FC) Network")
            if isinstance(self.action_space, gym.spaces.Discrete):
                print(f"Detected discrete action space, total number of actions: {self.action_space.n}")
                self.online_model = FCNetwork(self.observation_space.shape[0], self.action_space.n, hidden_space).to(device)
                self.target_model = FCNetwork(self.observation_space.shape[0], self.action_space.n, hidden_space).to(device)
            else:
                raise ValueError(f"Detected non-discrete action space, this class works only with discrete action space problems!")
        # 2. Atari games, observation space is a Box and shape is 3D
        elif isinstance(self.observation_space, gym.spaces.Box) and len(self.observation_space.shape) == 3:
            print(f"Detected a classic environment, observation space shape: {self.observation_space.shape}, using Convolutional Neural Network (CNN)")
            if isinstance(self.action_space, gym.spaces.Discrete):
                print(f"Detected discrete action space, total number of actions: {self.action_space.n}")
                self.online_model = CNNNetwork(self.observation_space.shape, self.action_space.n, hidden_space).to(device)
                self.target_model = CNNNetwork(self.observation_space.shape, self.action_space.n, hidden_space).to(device)
            else:
                raise ValueError(f"Detected non-discrete action space, this class works only with discrete action space problems!")
        else:
            raise ValueError(f"Detected unsupported observation space: {self.observation_space}. This class works only with gym.spaces.Box spaces that have shape of either 1D (vectors) or 3D (images)")

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

    def save_model(self, env_name):
        save_path = f"results/{env_name}"
        os.makedirs(save_path, exist_ok=True)
        online_model_path = os.path.join(save_path, "online_model.pth")
        target_model_path = os.path.join(save_path, "target_model.pth")
        torch.save(self.online_model.state_dict(), online_model_path)
        torch.save(self.target_model.state_dict(), target_model_path)
    
    def load_model(self, env_name):
        load_path = f"results/{env_name}"
        os.makedirs(load_path, exist_ok=True)
        online_model_path = os.path.join(load_path, "online_model.pth")
        target_model_path = os.path.join(load_path, "target_model.pth")
        self.online_model.load_state_dict(torch.load(online_model_path))
        self.target_model.load_state_dict(torch.load(target_model_path))

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def train(self, mean_rewards: List, std_rewards: List, max_episodes: int = 100000, mean_n_episodes: int = 50):
        pass