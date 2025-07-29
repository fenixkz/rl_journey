import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import os

class Agent(nn.Module):
    '''
    This is our Agent with a policy represented by a neural network with three layers. The network accepts the state as the input and outputs a probability distribution over actions. 
    '''
    def __init__(self, obs_dim, act_dim, device):
        super(Agent, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

        self.device = device
        self.relu = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.policy(x)

    def choose_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        '''
        A function that decides what action to choose given the state. Can be deterministic (choosing action with the highest probability) or stochastic (sampling action from a probability distribution). 
        '''
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) # Add a batch dimension
        # Forward pass, get logits
        logits = self.forward(state)
        if deterministic: # If determinstic, then just return the index of action corresponding to the maximum logit
            return logits.argmax(dim=-1).item()
        # Otherwise sample
        probs = F.softmax(logits, dim=-1) # Convert logits to probabilities
        dist = Categorical(probs)         # Create a distribution object
        action = dist.sample()            # Sample an action
        # Return as numpy scalar or array (remove batch dim)
        return action.item()


    def learn(self, states, actions):
        '''
        A function that trains the network to better predict actions from given states. 
        Uses CrossEntropyLoss
        '''
        # Convert to tensors 
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        # Get predictions
        pred_actions = self.forward(states_tensor)
        # Backward pass
        self.optimizer.zero_grad()
        loss = self.loss(pred_actions, actions_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()

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
