import numpy as np
import torch
import torch.nn as nn

# --- Fully Connected Versions (for vector-based envs) ---


class Actor(nn.Module):
    """
    Actor network for environments with vector-based observations.
    """

    def __init__(self, obs_space, action_space, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc_out = nn.Linear(hidden_size // 4, action_space)
        # Change to tanh to experiment
        self.relu = nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.fc_out.weight, gain=0.01)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, state: np.ndarray):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        logits = self.fc_out(x)
        return logits


class Critic(nn.Module):
    """
    Critic network for environments with vector-based observations.
    """

    def __init__(self, obs_space, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc_out = nn.Linear(hidden_size // 4, 1)
        self.relu = nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.fc_out.weight, gain=1.0)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, state: np.ndarray):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        value = self.fc_out(x)
        return value


class ActorCriticFC(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=64):
        super(ActorCriticFC, self).__init__()
        self.actor = Actor(obs_space, action_space, hidden_size)
        self.critic = Critic(obs_space, hidden_size)

    def forward(self, state):
        logits = self.actor(state)
        value = self.critic(state)
        return logits, value

    def get_value(self, state):
        return self.critic(state)


class ActorCriticCNN(nn.Module):
    """
    A unified Actor-Critic network for Atari with a shared backbone.
    """

    def __init__(self, obs_space_shape, action_space, hidden_size=512):
        super(ActorCriticCNN, self).__init__()
        in_channels = obs_space_shape[0]

        # Shared CNN backbone for feature extraction
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_space_shape)
            feature_size = self.backbone(dummy_input).shape[1]

        # --- Actor Head ---
        self.actor_head = nn.Sequential(
            nn.Linear(feature_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, action_space)
        )

        # --- Critic Head ---
        self.critic_head = nn.Sequential(nn.Linear(feature_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_weights()

    def _init_weights(self):
        # Initialize backbone weights
        for layer in self.backbone:
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

        # Initialize actor head weights
        actor_fc_layer = self.actor_head[0]
        nn.init.orthogonal_(actor_fc_layer.weight, gain=np.sqrt(2))
        nn.init.constant_(actor_fc_layer.bias, 0)
        actor_out_layer = self.actor_head[2]
        nn.init.orthogonal_(actor_out_layer.weight, gain=0.01)
        nn.init.constant_(actor_out_layer.bias, 0)

        # Initialize critic head weights
        critic_fc_layer = self.critic_head[0]
        nn.init.orthogonal_(critic_fc_layer.weight, gain=np.sqrt(2))
        nn.init.constant_(critic_fc_layer.bias, 0)
        critic_out_layer = self.critic_head[2]
        nn.init.orthogonal_(critic_out_layer.weight, gain=1.0)
        nn.init.constant_(critic_out_layer.bias, 0)

    def forward(self, state: np.ndarray):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if state.ndim == 3:
            state = state.unsqueeze(0)

        # Pass through the shared backbone ONCE
        features = self.backbone(state)

        # Get actor logits and critic value
        logits = self.actor_head(features)
        value = self.critic_head(features)

        return logits, value

    def get_value(self, state: np.ndarray):
        """
        Computes the value V(s) without computing the actor's logits.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if state.ndim == 3:
            state = state.unsqueeze(0)

        # Pass through the shared backbone ONCE
        features = self.backbone(state)

        # Get critic value ONLY
        value = self.critic_head(features)

        return value
