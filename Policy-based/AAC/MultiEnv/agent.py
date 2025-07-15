import torch
import torch.nn as nn
import torch.nn.functional as F



class Actor(nn.Module):

    def __init__(self, obs_space, action_space, hidden_size=64, num_layers=2, dropout_rate=0.0):
        super(Actor, self).__init__()
        
        self.num_layers = num_layers
        
        if num_layers == 1:
            # Single hidden layer (original architecture)
            self.fc1 = nn.Linear(obs_space, hidden_size)
            self.fc_out = nn.Linear(hidden_size, action_space)
        elif num_layers == 2:
            # Two hidden layers for more complex environments
            self.fc1 = nn.Linear(obs_space, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc_out = nn.Linear(hidden_size, action_space)
        else:
            # Three hidden layers for very complex environments
            self.fc1 = nn.Linear(obs_space, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc_out = nn.Linear(hidden_size // 2, action_space)
        
        self.relu = nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, state):
        # Ensure input is a 2D tensor (batch_size, obs_space)
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        x = self.relu(self.fc1(state))
        x = self.dropout(x) if self.dropout else x
        if self.num_layers >= 2:
            x = self.relu(self.fc2(x))
        x = self.dropout(x) if self.dropout else x
        if self.num_layers >= 3:
            x = self.relu(self.fc3(x))
        x = self.dropout(x) if self.dropout else x  
        x = self.fc_out(x)
        return x
    

class Critic(nn.Module):
    def __init__(self, obs_space, hidden_size=64, num_layers=2, dropout_rate=0.0):
        super(Critic, self).__init__()
        
        self.num_layers = num_layers
        
        if num_layers == 1:
            # Single hidden layer (original architecture)
            self.fc1 = nn.Linear(obs_space, hidden_size)
            self.fc_out = nn.Linear(hidden_size, 1)
        elif num_layers == 2:
            # Two hidden layers for more complex environments
            self.fc1 = nn.Linear(obs_space, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc_out = nn.Linear(hidden_size, 1)
        else:
            # Three hidden layers for very complex environments
            self.fc1 = nn.Linear(obs_space, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc_out = nn.Linear(hidden_size // 2, 1)
        
        self.relu = nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, state):
        # Ensure input is a 2D tensor (batch_size, obs_space)
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        x = self.relu(self.fc1(state))
        x = self.dropout(x) if self.dropout else x
        if self.num_layers >= 2:
            x = self.relu(self.fc2(x))
        x = self.dropout(x) if self.dropout else x
        if self.num_layers >= 3:
            x = self.relu(self.fc3(x))
        x = self.dropout(x) if self.dropout else x  
        x = self.fc_out(x)
        return x
    

class Agent(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=64, num_layers=2, dropout_rate=0.0):
        super(Agent, self).__init__()
        self.actor = Actor(obs_space, action_space, hidden_size, num_layers, dropout_rate)
        self.critic = Critic(obs_space, hidden_size, num_layers, dropout_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
    
