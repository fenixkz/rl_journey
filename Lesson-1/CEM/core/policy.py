import torch
import torch.nn as nn


class FCNetwork(nn.Module):
    """
    A fully-connected neural network. This is our policy, pi(theta).
    Given the state, returns the probability distribution over actions.
    """

    def __init__(self, obs_space: int, action_space: int, hidden_space: int, device: str):
        super(FCNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, hidden_space // 2),
            nn.ReLU(),
            nn.Linear(hidden_space // 2, action_space),
        )
        self._init_weights()
        self.device = device

    def forward(self, observation: torch.Tensor):
        return self.network(observation)

    def _init_weights(self):
        """Initialize weights using Xavier initialization for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)
