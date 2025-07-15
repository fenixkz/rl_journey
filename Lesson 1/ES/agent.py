import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import time
import os
import sys
from tqdm import tqdm

# --- Agent Definition ---
# This Agent is simpler than the CEM one. It just holds the policy network.
# The "learning" happens in the main loop by directly manipulating its weights.
class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super(Agent, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, x):
        return self.policy(x)

    def choose_action(self, state: np.ndarray) -> np.ndarray:
        """Chooses an action based on the policy's output logits."""
        state_t = torch.tensor(state, dtype=torch.float32)
        logits = self.forward(state_t)
        # Return the action with the highest logit (deterministic)
        return logits.argmax(dim=-1).item()

    def get_weights(self):
        """Helper function to get network weights as a single flat vector."""
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_weights(self, weights_flat):
        """Helper function to set network weights from a single flat vector."""
        offset = 0
        for param in self.parameters():
            param_shape = param.data.shape
            num_elements = param.data.numel()
            # Reshape the flat vector segment and assign it to the parameter
            param.data.copy_(weights_flat[offset:offset + num_elements].reshape(param_shape))
            offset += num_elements

    def save_policy(self, env_name: str):
        save_path = f"results/{env_name}"
        os.makedirs(save_path, exist_ok=True)
        print("--- Saving Policy ---")
        torch.save(self.state_dict(), f"{save_path}/policy.pth")
        print(f"Policy saved in {save_path}")

    def load_policy(self, env_name: str):
        load_path = f"results/{env_name}"
        try:
            self.load_state_dict(
                torch.load(f"{load_path}/policy.pth")
            )
        except FileNotFoundError:
            print(f"Policy file not found in {load_path}. Using untrained policy.")
