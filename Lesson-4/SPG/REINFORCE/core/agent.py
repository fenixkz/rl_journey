import os
import sys
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from core.policy import REINFORCE
from tqdm import tqdm

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../../../")
sys.path.append(os.path.abspath(parent_path))

from common.base_agent import BaseAgent  # noqa: E402


class REINFORCEAgent(BaseAgent):

    def __init__(
        self,
        env_id: str,
        solved_threshold: int,
        gamma: float = 0.99,
        hidden_size: int = 64,
        lr: float = 3e-4,
        seed: int = 24,
        use_normalization: bool = False,
        use_entropy: bool = False,
        entropy_coef: float = 1e-2,
    ):
        # Create the environment
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        assert len(env.observation_space.shape) == 1, "REINFORCE supports for now only Box2D envs"

        # Initialize the base class
        super().__init__(env=env, solved_threshold=solved_threshold, seed=seed)

        # --- Hyperparameters ---
        self.gamma = gamma
        self.use_normalization = use_normalization
        self.use_entropy = use_entropy
        self.entropy_coef = entropy_coef

        # --- Policy ---
        self.policy = REINFORCE(env.observation_space.shape[0], env.action_space.n, hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def choose_action(self, state: np.ndarray) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        # State is internally converted to a tensor by the agent
        # Logits are the raw scores for each action
        logits = self.policy(state)  # shape: (batch_size, action_space)

        # Apply softmax to get probabilities
        # Convert logits to probabilities via softmax operator
        probs = F.softmax(logits, dim=-1)  # shape: (batch_size, action_space)

        # Create a categorical distribution from the probabilities
        # This distribution will be used to sample actions and compute log probabilities
        action_dist = torch.distributions.Categorical(
            probs
        )  # Categorical distribution for sampling actions, shape: (batch_size, action_space)

        # Sample an action from the distribution
        # action_dist.sample() returns a tensor of shape (batch_size,) with the sampled actions
        # We use item() to get the scalar value from the tensor
        action = action_dist.sample()  # Sample an action from the distribution, shape: (batch_size,)
        return action_dist, action

    def play_one_episode(self):
        state, _ = self.env.reset()
        done = False

        # Data structures to store log probabilities, entropies, and rewards
        log_probs = []
        entropies = []
        rewards = []

        while not done:
            action_dist, action = self.choose_action(state)
            # Apply the action to the environment
            next_state, reward, terminated, truncated, _ = self.env.step(
                action.item()
            )  # Use .item() to extract the integer from the tensor
            done = terminated or truncated

            if self.use_entropy:
                # Use entropy to encourage exploration
                # Entropy is calculated as sum of -p * log(p) for each action probability
                # but we can use the built-in method
                # # action_dist.entropy() has shape (batch_size,)
                entropies.append(action_dist.entropy())

            # Store log probabilities and rewards
            log_probs.append(
                action_dist.log_prob(action)
            )  # We can also use built-in method to compute log probabilities
            rewards.append(reward)  # Store the reward for this step
            state = next_state  # Transition to the next state

        return log_probs, rewards, entropies

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        returns = []
        G = 0
        # Use backward pass to compute returns for each step
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        # Convert to tensors
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def train(self, max_episodes: int = 2000):
        pbar = tqdm(range(max_episodes), desc="Training")
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
                loss -= (
                    self.entropy_coef * entropies.sum()
                )  # sum or mean? sum is more common in literature, but mean is also used

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Compute metrics and check if environment is solved
            self.train_rewards.append(sum(rewards))
            avg_reward = np.mean(self.train_rewards[-100:])

            if avg_reward >= self.solved_threshold:
                pbar.write(f"🎉 Environment solved at episode {e}!")
                break
            pbar.set_postfix(
                {
                    "Mean": f"{avg_reward:.1f}",
                }
            )

        self.env.close()
        pbar.close()

    def save(self, save_path: str):
        """
        Save policy for later evaluation
        """
        os.makedirs(save_path, exist_ok=True)
        policy_save_path = os.path.join(save_path, "policy.pth")
        torch.save(self.policy.state_dict(), policy_save_path)

    def load(self, load_path: str):
        """
        Load pre-trained policy
        """
        os.makedirs(load_path, exist_ok=True)
        policy_save_path = os.path.join(load_path, "policy.pth")
        self.policy.load_state_dict(torch.load(policy_save_path))
