import torch
import torch.nn as nn
import torch.nn.functional as F


class REINFORCE(nn.Module):
    '''
    Three layer FC network 
    '''
    def __init__(self, obs_space, action_space, hidden_size=64):
        super(REINFORCE, self).__init__()
        self.fc1 = nn.Linear(obs_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_out = nn.Linear(hidden_size // 2, action_space)
        
        self.relu = nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        # Ensure input is a 2D tensor (batch_size, obs_space)
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        x = self.relu(self.fc1(state))
        
        x = self.relu(self.fc2(x))
        
        x = self.relu(self.fc3(x))
        x = self.fc_out(x)
        return x
