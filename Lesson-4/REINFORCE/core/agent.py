import torch
import torch.nn as nn
import torch.nn.functional as F
from core.policy import REINFORCE
import gymnasium as gym
import numpy as np
import random 
from tqdm import tqdm
from collections import deque
from typing import List, Tuple
import os 

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

class REINFORCEAgent:

    def __init__(self, 
                 env: gym.Env,
                 solved_threshold: int,
                 gamma: float = 0.99,
                 hidden_size: int = 64,
                 lr: float = 3e-4,
                 seed: int = 24,
                 use_normalization: bool = False,
                 use_entropy: bool = False,
                 entropy_coef: float = 1e-2):
        assert len(env.observation_space.shape) == 1, "REINFORCE supports for now only Box2D envs"
        self.env = env
        self.solved_threshold = solved_threshold
        self.gamma = gamma
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.use_normalization = use_normalization
        self.use_entropy = use_entropy
        self.entropy_coef = entropy_coef
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        set_seed(seed)
        self.policy = REINFORCE(self.observation_space.shape[0], self.action_space.n, hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        env.reset(seed = seed)
        env.action_space.seed(seed)
    
    def act(self, state: np.ndarray) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        # State is internally converted to a tensor by the agent
        # Logits are the raw scores for each action
        logits = self.policy(state) # shape: (batch_size, action_space)
        
        # Apply softmax to get probabilities
        # Convert logits to probabilities via softmax operator
        probs = F.softmax(logits, dim = -1) # shape: (batch_size, action_space)
        
        # Create a categorical distribution from the probabilities
        # This distribution will be used to sample actions and compute log probabilities
        action_dist = torch.distributions.Categorical(probs) # Categorical distribution for sampling actions, shape: (batch_size, action_space)

        # Sample an action from the distribution
        # action_dist.sample() returns a tensor of shape (batch_size,) with the sampled actions
        # We use item() to get the scalar value from the tensor
        action = action_dist.sample() # Sample an action from the distribution, shape: (batch_size,)
        return action_dist, action 
    
    def play_one_episode(self):
        state, _ = self.env.reset()
        done = False
        
        # Data structures to store log probabilities, entropies, and rewards
        log_probs = []
        entropies = []
        rewards = []

        while not done:
            action_dist, action = self.act(state)
            # Apply the action to the environment
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())  # Use .item() to extract the integer from the tensor
            done = terminated or truncated
            
            if self.use_entropy:
                # Use entropy to encourage exploration
                # Entropy is calculated as sum of -p * log(p) for each action probability, but we can use the built-in method
                entropies.append(action_dist.entropy()) # action_dist.entropy() has shape (batch_size,)
            
            # Store log probabilities and rewards
            log_probs.append(action_dist.log_prob(action)) # We can also use built-in method to compute log probabilities 
            rewards.append(reward) # Store the reward for this step
            state = next_state # Transition to the next state

        return log_probs, rewards, entropies

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards): # Use backward pass to compute returns for each step
            G = r + self.gamma * G
            returns.insert(0, G)
        
        # Convert to tensors
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def train(self, all_rewards: List, max_episodes: int = 2000):
        rewards_log = deque(maxlen=100)
        pbar = tqdm(range(max_episodes), desc=f"Training")
        for e in pbar:
            log_probs, rewards, entropies = self.play_one_episode()

            returns = self.compute_returns(rewards)
            log_probs = torch.stack(log_probs).to(self.device)
            
            # Normalize returns if specified
            # Normalization helps to stabilize the training by reducing the huge variation in returns
            if self.use_normalization:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Compute the loss 
            # The loss is the negative log probability of the actions taken, weighted by the returns
            loss = -(log_probs * returns).sum()
            
            if self.use_entropy:
                entropies = torch.stack(entropies).to(self.device)
                loss -= self.entropy_coef * entropies.sum() # sum or mean? sum is more common in literature, but mean is also used
  
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute metrics and check if environment is solved
            avg_reward = -float('inf')
            if len(rewards_log) > 10: avg_reward = np.mean(rewards_log) # to avoid computing mean of empty deque
            
            rewards_log.append(sum(rewards))
            all_rewards.append(sum(rewards))
            
            if avg_reward >= self.solved_threshold:
                pbar.write(f"🎉 Environment solved at episode {e}!")
                break  # Stop training if solved
            # Print progress
            pbar.set_postfix({
                                'Mean': f"{avg_reward:.1f}",
                            })

        self.env.close()
        pbar.close()
    
    def save_policy(self, save_path: str):
        '''
        Save policy for later evaluation
        '''
        os.makedirs(save_path, exist_ok=True)
        policy_save_path = os.path.join(save_path, "policy.pth")
        torch.save(self.policy.state_dict(), policy_save_path)
    
    def load_policy(self, load_path: str):
        '''
        Load pre-trained policy
        '''
        os.makedirs(load_path, exist_ok=True)
        policy_save_path = os.path.join(load_path, "online_model.pth")
        self.policy.load_state_dict(torch.load(policy_save_path))
