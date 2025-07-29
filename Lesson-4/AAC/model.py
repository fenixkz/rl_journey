import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, obs_space, action_space):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_space, 64)
        self.fc2 = nn.Linear(64, action_space)
        self.relu = nn.ReLU()

    def forward(self, state):
        if state.ndim == 1:
            state = state.unsqueeze(0) # Add a batch dimension if not present
        x = self.relu(self.fc1(state))
        x = self.fc2(x)
        return x # Returns logits, shape [batch_size, action_space]
    

class Critic(nn.Module):
    def __init__(self, obs_space):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_space, 64)
        self.fc2 = nn.Linear(64, 1) # One output for estimating V(s)
        self.relu = nn.ReLU()

    def forward(self, state):
        if state.ndim == 1:
            state = state.unsqueeze(0) # Add a batch dimension if not present
        x = self.relu(self.fc1(state))
        x = self.fc2(x)
        return x # Returns V(s), shape [batch_size, 1]
    

class Agent(nn.Module):
    def __init__(self, obs_space, action_space, device):
        super(Agent, self).__init__()
        self.actor = Actor(obs_space, action_space)
        self.critic = Critic(obs_space)
        self.device = device
        
    def act(self, state):
        '''
        Given a state, return the action probabilities. 
        $ \pi(a | s)$
        '''
        if not isinstance(state, torch.Tensor): # Ensure state is a tensor
            state = torch.FloatTensor(state).to(self.device)
        logits = self.actor(state) # logits shape [batch_size, action_space]
        return F.softmax(logits, dim=-1)  # Return action probabilities

    def evaluate(self, state):
        '''
        Compute the state value function $V(s)$
        '''
        if not isinstance(state, torch.Tensor): # Ensure state is a tensor
            state = torch.FloatTensor(state).to(self.device)
        value = self.critic(state) # Value shape: [1, 1]
        return value
    
