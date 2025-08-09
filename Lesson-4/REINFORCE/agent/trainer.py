import torch
import torch.nn as nn
import torch.nn.functional as F
from policy import REINFORCE
import gymnasium as gym
import numpy as np
import random 
from tqdm import tqdm
from collections import deque
from typing import List

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

class Trainer:

    def __init__(self, 
                 env: gym.Env,
                 solved_threshold: int,
                 hidden_size: int = 64,
                 lr: float = 3e-4,
                 seed: int = 24,
                 use_normalize: bool = False,
                 use_entropy: bool = False):
        assert len(self.observation_space.shape) > 1, "REINFORCE supports for now only Box2D envs"
        self.env = env
        self.solved_threshold = solved_threshold
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.use_normalize = use_normalize
        self.use_entropy = use_entropy
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        set_seed(seed)
        self.policy = REINFORCE(self.observation_space.shape[0], self.action_space.n, hidden_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        env.reset(seed = seed)
        env.action_space.seed(seed)

    def train(self, mean_rewards: List, std_rewards: List, max_episodes: int = 2000, mean_n_episodes: int = 50):
        rewards_log = deque(maxlen=mean_n_episodes)
        best_avg_reward = float('-inf')
        solved_episode = None
        pbar = tqdm(range(max_episodes), desc=f"Training")
        for e in pbar:
            state, _ = self.env.reset()
            done = False
            
            # Data structures to store log probabilities, entropies, and rewards
            log_probs = []
            entropies = []
            rewards = []

            while not done:
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
                # Apply the action to the environment
                next_state, reward, terminated, truncated, _ = self.env.step(action.item()) # Use .item() to extract the integer from the tensor
                done = terminated or truncated
                
                if self.use_entropy:
                    # Use entropy to encourage exploration
                    # Entropy is calculated as sum of -p * log(p) for each action probability, but we can use the built-in method
                    entropies.append(action_dist.entropy()) # action_dist.entropy() has shape (batch_size,)
                
                # Store log probabilities and rewards
                log_probs.append(action_dist.log_prob(action)) # We can also use built-in method to compute log probabilities 
                rewards.append(reward) # Store the reward for this step
                state = next_state # Transition to the next state

            # Compute returns
            returns = []
            G = 0
            for r in reversed(rewards):
                # Use backward pass to compute returns for each step
                G = r + 0.99 * G
                returns.insert(0, G)
            
            # Convert to tensors
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            log_probs = torch.stack(log_probs).to(self.device)
            
            # Normalize returns if specified
            # Normalization helps to stabilize the training by reducing the huge variation in returns
            if self.use_normalize:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Compute the loss 
            # The loss is the negative log probability of the actions taken, weighted by the returns
            # This is the REINFORCE algorithm
            # We want to maximize the expected return, so we minimize the negative log probability
            loss = -(log_probs * returns).sum()
            
            if self.use_entropy:
                entropies = torch.stack(entropies).to(self.device)
                loss -= 0.01 * entropies.sum() # sum or mean? sum is more common in literature, but mean is also used

            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Check if environment is solved
            avg_reward = np.mean(rewards_log)
            rewards_log.append(sum(rewards))
            if len(rewards_log) == mean_n_episodes:
                mean_rewards.append(avg_reward)
                std_rewards.append(np.std(rewards_log))

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
            
            if avg_reward >= self.solved_threshold and solved_episode is None:
                solved_episode = e
                pbar.write(f"🎉 Environment solved at episode {e}! Average reward: {avg_reward:.2f}")
                break  # Stop training if solved
            # Print progress
            pbar.set_postfix({
                                'Mean': f"{avg_reward:.1f}",
                            })

        self.env.close()
        pbar.close()
        