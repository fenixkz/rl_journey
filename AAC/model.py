import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):

    def __init__(self, obs_space, action_space):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(obs_space, 64)
        self.fc2 = nn.Linear(64, action_space)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.fc2(x)
        return x
    

