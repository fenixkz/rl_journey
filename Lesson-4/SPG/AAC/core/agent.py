import os
import sys
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from core.actor_critic import Actor, Critic
from tqdm import tqdm

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../../../")
sys.path.append(os.path.abspath(parent_path))

from common.base_agent import BaseAgent  # noqa: E402


class AACAgent(BaseAgent):
    """
    Advantage Actor-Critic (AAC) with Generalized Advantage Estimate (GAE)
    """

    def __init__(
        self,
        env_id: str,
        solved_threshold: int,
        gamma: float = 0.99,
        hidden_size: int = 64,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        seed: int = 24,
        use_normalization: bool = False,
        use_entropy: bool = False,
        entropy_coef: float = 0.01,
        num_steps: int = 5,
        gae_lambda: float = 0.95,
        tau: float = 0.05,
    ):

        # Create the environment
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        assert len(env.observation_space.shape) == 1, "AAC supports for now only Box2D envs"

        # Initialize the base class
        super().__init__(env=env, solved_threshold=solved_threshold, seed=seed)

        # --- Hyperparameters ---
        self.gamma = gamma
        self.use_normalization = use_normalization
        self.use_entropy = use_entropy
        self.entropy_coef = entropy_coef
        self.num_steps = num_steps
        self.gae_lambda = gae_lambda
        self.tau = tau

        # --- Policy: Actor, Critic --
        self.actor = Actor(env.observation_space.shape[0], env.action_space.n, hidden_size).to(self.device)
        self.online_critic = Critic(env.observation_space.shape[0], hidden_size).to(self.device)
        self.target_critic = Critic(env.observation_space.shape[0], hidden_size).to(self.device)
        # Copy weights
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.online_critic.parameters(), lr=critic_lr)
        # A placeholder to track the current state in between rollouts
        self.current_state = None

    def soft_update_target_network(self):
        """
        Soft target network update
        """
        target_net_state_dict = self.target_critic.state_dict()
        online_net_state_dict = self.online_critic.state_dict()
        for key in online_net_state_dict:
            target_net_state_dict[key] = online_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                1 - self.tau
            )
        self.target_critic.load_state_dict(target_net_state_dict)

    def choose_action(self, state: np.ndarray) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        logits = self.actor(state)
        probs = F.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action_dist, action

    def evaluate(self, state: np.ndarray) -> torch.Tensor:
        return self.online_critic(state)

    @torch.no_grad()
    def compute_returns_and_advantages(
        self, rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using GAE from collected rollout data.
        Args:
            next_value: Value for the states after the last rollout step.
        Returns:
            returns: N-step returns (targets for the critic).
            advantages: GAE advantages (weights for the actor).
        """
        # Initialize advantages, shape (N, 1)
        advantages = torch.zeros_like(values).to(self.device)
        last_gae_lambda = 0

        # Iterate in reverse order
        for t in reversed(range(self.num_steps)):
            # The last element should have next_values computed explicitly
            if t == self.num_steps - 1:
                next_vals = next_value
            else:  # Or get the stored values of next index
                next_vals = values[t + 1]
            # If the done is True, then it means that the next state is terminal
            next_non_terminal = 1.0 - dones[t]
            # Calculate deltas (td error)
            delta = rewards[t] + self.gamma * next_vals * next_non_terminal - values[t]
            # Calculate advantage using GAE
            advantages[t] = last_gae_lambda = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda

        # Total return is Q = A + V
        returns = advantages + values
        return returns, advantages

    def collect_rollout(self):
        """
        Collects `num_steps` of experience from environment.
        """
        # Query the state from the last rollout
        state = self.current_state

        # Lists to store the N step data
        values, logprobs, entropies, rewards, dones = [], [], [], [], []
        for _ in range(self.num_steps):
            # Get action from the actor given the state
            action_dist, action = self.choose_action(state)
            # Get the value of this state
            state_value = self.evaluate(state)
            # Step in the environments
            next_state, reward, terminated, truncated, info = self.env.step(action.item())

            # Store data
            logprobs.append(action_dist.log_prob(action))  # Gradient tracked
            values.append(state_value.squeeze())  # Gradient tracked
            entropies.append(action_dist.entropy())  # Gradient tracked
            rewards.append(reward)  # A scalar
            # Store only if terminated
            dones.append(terminated)  # A scalar

            if terminated or truncated:
                if "episode" in info:
                    self.train_rewards.append(info["episode"]["r"])
                state, _ = self.env.reset()
            else:
                state = next_state

        # Convert data that does NOT need gradients
        # Ensure tensors are 1D (using .squeeze() if they come from envs with extra dims)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).squeeze()
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device).squeeze()

        # Stack tensors that DO need gradients.
        # torch.stack is a differentiable operation that preserves the computation graph.
        logprobs = torch.stack(logprobs)
        values = torch.stack(values)
        entropies = torch.stack(entropies)

        # Save current state for the future rollout
        self.current_state = state

        # Find the value of the final state using the target critic
        with torch.no_grad():
            next_value = self.target_critic(state).squeeze()

        # Calculate returns and advantages
        returns, advantages = self.compute_returns_and_advantages(rewards, dones, values, next_value)

        return logprobs, values, entropies, returns, advantages

    def learn(
        self,
        logprobs: torch.Tensor,
        values: torch.Tensor,
        entropies: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ):
        # Normalize advantage if specified
        if self.use_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute losses
        # Actor loss: -log_prob * A(s, a).
        # Advantage is detached for actor loss, the gradients must flow only through log_probs
        actor_loss = -(logprobs.squeeze() * advantages.detach()).mean()

        # Add entropy to the loss, if specified
        if self.use_entropy:
            entropy_loss = self.entropy_coef * entropies.mean()
            actor_loss -= entropy_loss

        # Critic loss: L2 between target and state values
        critic_loss = F.mse_loss(values, returns)
        total_loss = actor_loss + critic_loss

        # Zero out the gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        total_loss.backward()

        # Clip the gradients for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(self.online_critic.parameters(), max_norm=1)

        # Step the optimizer
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def train(self, max_steps: int = 10000):
        pbar = tqdm(range(max_steps), desc="Training")

        state, _ = self.env.reset()

        # In the beginning set the initial state
        self.current_state = state
        avg_reward = -float("inf")

        for e in pbar:
            logprobs, values, entropies, returns, advantages = self.collect_rollout()
            self.learn(logprobs, values, entropies, returns, advantages)
            self.soft_update_target_network()

            if self.train_rewards:
                avg_reward = np.mean(self.train_rewards[-100:])

                if avg_reward >= self.solved_threshold:
                    pbar.write(f"🎉 Environment solved at episode {e}!")
                    break

            # Print progress
            pbar.set_postfix(
                {
                    "Episode": f"{len(self.train_rewards)}",
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
        actor_save_path = os.path.join(save_path, "actor.pth")
        critic_save_path = os.path.join(save_path, "critic.pth")
        torch.save(self.actor.state_dict(), actor_save_path)
        torch.save(self.online_critic.state_dict(), critic_save_path)

    def load(self, load_path: str):
        """
        Load pre-trained policy
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Load path does not exist: {load_path}")

        actor_save_path = os.path.join(load_path, "actor.pth")
        critic_save_path = os.path.join(load_path, "critic.pth")

        if not os.path.exists(actor_save_path):
            raise FileNotFoundError(f"Actor checkpoint not found: {actor_save_path}")
        if not os.path.exists(critic_save_path):
            raise FileNotFoundError(f"Critic checkpoint not found: {critic_save_path}")

        self.actor.load_state_dict(torch.load(actor_save_path))
        self.online_critic.load_state_dict(torch.load(critic_save_path))
