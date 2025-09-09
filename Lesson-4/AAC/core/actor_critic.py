import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):

    def __init__(self, obs_space, action_space, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_space)
        self.relu = nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state: np.ndarray):
        # Ensure input is a 2D tensor (batch_size, obs_space)
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        x = self.relu(self.fc1(state))
        x = self.fc2(x)
        return x  # Returns logits, shape [batch_size, action_space]


class Critic(nn.Module):
    def __init__(self, obs_space, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # One output for estimating V(s)
        self.relu = nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state: np.ndarray):
        # Ensure input is a 2D tensor (batch_size, obs_space)
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        x = self.relu(self.fc1(state))
        x = self.fc2(x)
        return x  # Returns V(s), shape [batch_size, 1]
