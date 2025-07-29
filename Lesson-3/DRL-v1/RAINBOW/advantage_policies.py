import torch
import torch.nn as nn

class FCD3QN(nn.Module):
    def __init__(self, obs_space, action_space, hidden_space: int = 128):
        super(FCD3QN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_space, hidden_space),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space)
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, 1) # A single output V(s)
        )

    def forward(self, x: torch.Tensor):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
    
class CNND3QN(nn.Module):
    def __init__(self, obs_space, action_space, hidden_space: int = 128):
        super(CNND3QN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(obs_space[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )   
        # Pass a zero tensor to get the final flattened shape of the resulting tensor
        with torch.no_grad():
            feature_size = self.backbone(torch.zeros(1, *obs_space)).view(1, -1).size(1)
        
        self.advantage = nn.Sequential(
            nn.Linear(feature_size, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space)
        )
        
        self.value = nn.Sequential(
            nn.Linear(feature_size, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, 1) # A single output V(s)
        )

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
