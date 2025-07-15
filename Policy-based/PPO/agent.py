import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os 
from torch.distributions.categorical import Categorical
class Actor(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc_out = nn.Linear(hidden_size//4, action_space)
        self.tanh = nn.Tanh()
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.fc_out.weight, gain=0.01)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, state):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        x = self.tanh(self.fc1(state))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        logits = self.fc_out(x)
        return logits

class Critic(nn.Module):
    def __init__(self, obs_space, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc_out = nn.Linear(hidden_size//4, 1)
        self.tanh = nn.Tanh()
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.fc_out.weight, gain=1.0)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, state):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        x = self.tanh(self.fc1(state))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        value = self.fc_out(x)
        return value

class PPOAgent(nn.Module):
    def __init__(self, 
                 obs_space, 
                 action_space, 
                 hidden_size=64):
        super(PPOAgent, self).__init__()
        self.actor = Actor(obs_space, action_space, hidden_size)
        self.critic = Critic(obs_space, hidden_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def act_and_evaluate(self, state, actions = None):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        logits = self.actor(state)
        probs = Categorical(logits=logits)
        if actions is None:
            actions = probs.sample()
        return actions, probs.log_prob(actions), probs.entropy(), self.critic(state), logits, probs.probs
    
    def evaluate(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        values = self.critic(state)
        return values
    
    def save_models(self, env_name="test"):
        """Saves the state dictionaries of the online actor and critic."""
        os.makedirs(f"results/{env_name}/models", exist_ok=True)

        print("--- Saving models ---")
        torch.save(
            self.actor.state_dict(),
            f"results/{env_name}/models/{env_name}_actor.pth"
        )
        torch.save(
            self.critic.state_dict(),
            f"results/{env_name}/models/{env_name}_critic.pth"
        )

    def load_models(self, env_name="test"):
        """Loads the state dictionaries for the online actor and critic."""
        print("--- Loading models ---")
        self.actor.load_state_dict(
            torch.load(f"results/{env_name}/models/{env_name}_actor.pth")
        )
        self.critic.load_state_dict(
            torch.load(f"results/{env_name}/models/{env_name}_critic.pth")
        )