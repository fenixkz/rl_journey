import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import os
import gymnasium as gym
import random
from tqdm import tqdm
from typing import List
import sys

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class FCNetwork(nn.Module):
    '''
    A fully-connected neural network. This is our policy, $\pi(\theta)$. 
    Given the state, returns the probability distribution over actions.
    '''
    
    def __init__(self, obs_space, action_space, hidden_space: int = 128):
        super(FCNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space),
        )
        self._init_weights()
        
    def forward(self, x: torch.Tensor):
        return self.network(x)
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)


class CEMAgent(nn.Module):
    '''
    This is our Agent with a policy represented by a neural network. 
    The network accepts the state as the input and outputs a probability distribution over actions. 
    '''
    def __init__(self, 
                 env: gym.Env,
                 solved_threshold: float,
                 hidden_dim: int = 64,
                 lr: float = 1e-3,
                 seed: int = 224,):
        assert(isinstance(env.action_space, gym.spaces.Discrete)), "Detected non-discrete action space, this class works only with discrete action space problems!"
        set_seed(seed)
        super(CEMAgent, self).__init__()

        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.solved_threshold = solved_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.hidden_dim = hidden_dim


        self.policy = FCNetwork(self.observation_space.shape[0], self.action_space.n, hidden_dim).to(self.device)
        

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        
        # First reset for reproducing same results
        env.reset(seed = seed)
        env.action_space.seed(seed)

    def choose_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        '''
        A function that decides what action to choose given the state. Can be deterministic (choosing action with the highest probability) or stochastic (sampling action from a probability distribution). 
        '''
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device) # Add a batch dimension
        # Forward pass, get logits
        logits = self.policy.forward(obs)
        if deterministic: # If determinstic, then just return the index of action corresponding to the maximum logit
            return logits.argmax(dim=-1).item()
        # Otherwise sample
        probs = F.softmax(logits, dim=-1) # Convert logits to probabilities
        dist = Categorical(probs)         # Create a distribution object
        action = dist.sample()            # Sample an action
        # Return as numpy scalar or array (remove batch dim)
        return action.item()

    def select_elite_episodes(self, episodes_data, percentile):
        """
        Select elite episodes based on total episode rewards.
        episodes_data: list of (states, actions, total_reward) tuples
        """
        # Sort episodes by total reward
        episodes_data.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate how many episodes to keep
        n_elite = int(len(episodes_data) * (100 - percentile) / 100)
        n_elite = max(1, n_elite)  # Keep at least 1 episode
        
        # Get elite episodes
        elite_episodes = episodes_data[:n_elite]
        
        # Extract all states and actions from elite episodes
        elite_obs = []
        elite_actions = []
        
        for states, actions, _ in elite_episodes:
            elite_obs.extend(states)
            elite_actions.extend(actions)
        
        return np.array(elite_obs), np.array(elite_actions)

    def learn(self, observations, actions):
        '''
        A function that trains the network to better predict actions from given states. 
        Uses CrossEntropyLoss
        '''
        # Convert to tensors 
        obs_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        # Get predictions
        pred_actions = self.policy.forward(obs_tensor)
        # Backward pass
        self.optimizer.zero_grad()
        loss = self.loss(pred_actions, actions_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_policy(self, save_path: str):
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

    def play_one_episode(self, env: gym.Env = None, deterministic: bool = False, render: bool = False):
        if env is None:
            env = self.env
        obs, _ = env.reset()
        history = []
        done = False
        total_reward = 0
        while not done:
            action = self.choose_action(obs=obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            history.append((obs, action))
            if render: env.render()
            obs = next_obs
            total_reward += reward
        return history, total_reward
        
    def train(self, all_rewards: List, num_epochs: int, num_episodes: int, percentile: float):
        
        pbar = tqdm(range(num_epochs), desc="Training", postfix={"mean_reward": 0})

        for e in pbar:
            # Collect data from multiple episodes
            episodes_data = []

            # Sequential processing for classic environments
            # Plan N episodes and collect data
            for _ in range(num_episodes):
                history, total_reward = self.play_one_episode()

                # Append to the external list total reward
                all_rewards.append(total_reward)

                # Extract observations and actions from history
                episode_observations = [step[0] for step in history]
                episode_actions = [step[1] for step in history]

                # Store episode data: (states, actions, total_reward)
                episodes_data.append((episode_observations, episode_actions, total_reward))
            
            episodic_rewards = [episode[2] for episode in episodes_data]
            
            mean_reward = np.mean(episodic_rewards)
            pbar.set_postfix_str(f"mean_reward: {mean_reward:.2f}")

            # Select elite episodes based on total episode rewards
            elite_obs, elite_actions = self.select_elite_episodes(episodes_data = episodes_data,
                                                                  percentile = percentile)

            if len(elite_obs) > 0:  # Only train if we have elite data
                self.learn(elite_obs, elite_actions)

            # Check for early termination
            if mean_reward > self.solved_threshold:
                pbar.close()
                print(f"Terminated after {e+1} steps with mean reward {mean_reward:.2f}")
                break