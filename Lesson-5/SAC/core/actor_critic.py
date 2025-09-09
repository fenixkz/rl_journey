import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = 1e-6


class Actor(nn.Module):
    """
    Actor accepts state as input and returns a distribution over actions.
    This is a stochastic actor that outputs mean and log_std for a Gaussian distribution.
    """

    def __init__(self, state_dim, action_dim: int, upper: float, lower: float):
        super().__init__()
        # Shared layers
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 256)

        # Mean and log_std heads
        self.mean_head = nn.Linear(256, action_dim)

        # We estimate log(sigma) instead of sigma directly. This is a standard
        # trick to ensure the standard deviation is always positive, as sigma = exp(log_sigma).
        # Using an activation like ReLU on a direct sigma output can lead to "dead"
        # gradients if the output becomes negative, stalling learning. The exp()
        # function has a non-zero gradient everywhere, making training more stable
        self.log_std_head = nn.Linear(256, action_dim)

        # Store action bounds
        self.register_buffer(
            "action_range",
            torch.tensor(
                (upper - lower) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_midpoint",
            torch.tensor(
                (upper + lower) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state: torch.Tensor):
        # Cast to tensor, add batch dimension and move to the device
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        # Ensure state is at least 2D (for single state inputs)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        # Clean-RL soft clamping
        # 1. Squash into [-1; 1] range using tanh
        log_std = torch.tanh(log_std)
        log_std = LOG_SIG_MIN + 0.5 * (LOG_SIG_MAX - LOG_SIG_MIN) * (log_std + 1)
        # Hard clamping
        # log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        """
        Sample action from the policy distribution and compute log probability.
        Uses reparameterization trick (mean + std * N(0,1))
        """
        # Get the mu and sigma (log sigma)
        mean, log_std = self.forward(state)
        # Log sigma -> sigma
        std = log_std.exp()
        # Construct a normal distribution using the parameters from the actor
        normal = Normal(mean, std)
        # .rsample() internally applies reparametrization trick
        x_t = normal.rsample()
        # To ensure [-1;1] range
        y_t = torch.tanh(x_t)
        # Scale to action boundaries
        action = y_t * self.action_range + self.action_midpoint

        # Get the log_prob of that action
        log_prob = normal.log_prob(x_t)
        # Since the initial action sampled was x_t, but the action returned is action
        # We have to correct the log_prob via change of variables formula for probability distributions
        # p_y(y) = p_x(x) * |dx/dy|
        # y = tanh(x) * m + b
        # d(tanh(x)) / dx = 1 - tanh^2(x)
        # log(p_y) = log(p_x) + log(dx/dy) = log(p_x) - log(dy/dx)
        # epsilon to avoid log(0)
        log_prob -= torch.log(self.action_range * (1 - y_t.pow(2)) + epsilon)
        # In N-dimensional action space, the probability of picking all N actions
        # P = p_0 * p_1 * ... * p_N
        # or
        # log(P) = sum(log(p_i))
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_range + self.action_midpoint
        return action, log_prob, mean


class Critic(nn.Module):
    """
    Critic accepts state and action as input and returns Q-values of each state-action pair.
    SAC uses twin Q-networks similar to TD3, but they are defined separately here.
    """

    def __init__(self, state_dim, action_dim: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 1)
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Returns both Q1 and Q2 values for the given state-action pair.
        """
        # Cast to tensor, add batch dimension and move to the device
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        # Ensure state is at least 2D (for single state inputs)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        features = torch.concat((state, action), dim=1)
        q1_value = self.q1(features)
        q2_value = self.q2(features)
        return q1_value, q2_value
