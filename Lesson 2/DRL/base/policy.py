import torch
import torch.nn as nn

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
