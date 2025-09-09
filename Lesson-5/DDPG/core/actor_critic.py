import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Actor accepts state as input and returns a single action per action_dimension.
    """

    def __init__(self, state_dim, action_dim: int, upper: float, lower: float):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            # Output N actions per action_space
            nn.Linear(256, action_dim),
            # Squash the output to the [-1, 1]
            nn.Tanh(),
        )
        self.register_buffer(
            "range",
            torch.tensor(
                (upper - lower) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "midpoint",
            torch.tensor(
                (upper + lower) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state: torch.Tensor):
        # Cast to tensor, add batch dimension and move to the device
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # Given the output tensor of shape (B, N-actions)
        # and range shape: (1, N),
        # via broadcasting result is (B, N-actions)
        action = self.model(state) * self.range + self.midpoint
        return action


class Critic(nn.Module):
    """
    Critic accepts state and action as input and returns a Q-values of each state-action pair,
    Works only for vector observations.
    If the observation is images, then this architecture won't work.
    """

    def __init__(self, state_dim, action_dim: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Assuming state and action are Tensors with batch dimension
        """
        # Cast to tensor, add batch dimension and move to the device
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        features = torch.concat((state, action), dim=1)
        q_value = self.model(features)
        return q_value
