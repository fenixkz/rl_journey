import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class Agent(nn.Module):

    def __init__(self, obs_dim, act_dim, device):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.device = device
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_dim)
        self.relu = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self.eval()
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        logits = self.forward(obs_tensor)
        if deterministic:
            return logits.argmax(dim=-1).item()
        probs = F.softmax(logits, dim=-1) # Convert logits to probabilities
        dist = Categorical(probs)         # Create a distribution object
        action = dist.sample()            # Sample an action
        # Return as numpy scalar or array (remove batch dim)
        return action.item()


    def fit(self, states, actions):
        self.train()
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        self.optimizer.zero_grad()
        pred_actions = self.forward(states_tensor)
        loss = self.loss(pred_actions, actions_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()
