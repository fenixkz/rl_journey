import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os 

class Actor(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc_out = nn.Linear(hidden_size//4, action_space)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
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
        x = self.relu(self.fc1(state))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        logits = self.fc_out(x)
        return logits

class Critic(nn.Module):
    def __init__(self, obs_space, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc_out = nn.Linear(hidden_size//4, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
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
        x = self.relu(self.fc1(state))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        value = self.fc_out(x)
        return value

class A2CAgent(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=64, device='cpu', 
                 actor_lr: float = 1e-4, critic_lr: float = 3e-4, entropy_coef: float = 0.01):
        super(A2CAgent, self).__init__()
        self.actor = Actor(obs_space, action_space, hidden_size)
        self.critic = Critic(obs_space, hidden_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_criterion = nn.MSELoss()
        self.device = device
        self.entropy_coef = entropy_coef
    
    def act_and_evaluate(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        logits = self.actor(state)
        probs = F.softmax(logits, dim=-1)
        values = self.critic(state)
        return probs, values
    
    def evaluate(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        values = self.critic(state)
        return values
    
    def learn(self, values, returns, advantages, log_probs, entropies):
        # 1. Calculate critic's loss (prediction, target)
        critic_loss = self.critic_criterion(values, returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Calculate actor's loss
        # Policy Gradient Loss + Entropy Bonus
        # We want to MAXIMIZE entropy, so we MINIMIZE its negative.
        actor_loss = -(advantages.detach() * log_probs).mean() - self.entropy_coef * entropies.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    
    def save_models(self, env_name="ddpg_agent"):
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

    def load_models(self, env_name="ddpg_agent"):
        """Loads the state dictionaries for the online actor and critic."""
        print("--- Loading models ---")
        self.actor.load_state_dict(
            torch.load(f"results/{env_name}/models/{env_name}_actor.pth")
        )
        self.critic.load_state_dict(
            torch.load(f"results/{env_name}/models/{env_name}_critic.pth")
        )