import os
import sys
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from core.actor_critic import ActorCriticCNN, ActorCriticFC
from tqdm import tqdm

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../../")
sys.path.append(os.path.abspath(parent_path))

from common.base_agent import BaseAgent  # noqa: E402
from common.utils.atari_utils import get_atari_env  # noqa: E402


def print_tensor(name, tensor):
    print(f"{name}: {tensor}")
    print(f"{name} shape: {tensor.shape}")


class A2CAgent(BaseAgent):
    """
    Asynchronous Advantage Actor-Critic (A2C) with Generalized Advantage Estimate (GAE)
    """

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        solved_threshold: int,
        critic_scale: float = 0.5,
        is_atari: bool = False,
        gamma: float = 0.99,
        hidden_size: int = 64,
        lr: float = 3e-4,
        seed: int = 24,
        use_normalization: bool = False,
        use_entropy: bool = False,
        entropy_coef: float = 0.01,
        num_steps: int = 5,
        gae_lambda: float = 0.95,
        tau: float = 0.05,
    ):
        if is_atari:
            make = get_atari_env
        else:
            make = gym.make
        # Create the environment
        env_fns = [lambda: gym.wrappers.RecordEpisodeStatistics(make(env_id)) for _ in range(num_envs)]
        # Now env is not a single, but multiple envs
        env = gym.vector.SyncVectorEnv(env_fns)
        assert isinstance(
            env.single_action_space, gym.spaces.Discrete
        ), "Detected non-discrete action space, this class works only with discrete action space problems!"
        # Initialize the base class
        super().__init__(env=env, solved_threshold=solved_threshold, seed=seed)

        # --- Hyperparameters ---
        self.env_id = env_id
        self.num_envs = num_envs
        self.gamma = gamma
        self.use_normalization = use_normalization
        self.use_entropy = use_entropy
        self.entropy_coef = entropy_coef
        self.num_steps = num_steps
        self.gae_lambda = gae_lambda
        self.tau = tau
        self.is_atari = is_atari
        self.critic_scale = critic_scale

        # --- Policy: Actor, Critic --
        if not is_atari:
            self.actor_critic = ActorCriticFC(
                env.single_observation_space.shape[0], env.single_action_space.n, hidden_size
            ).to(self.device)
            self.target_critic = ActorCriticFC(
                env.single_observation_space.shape[0], env.single_action_space.n, hidden_size
            ).to(self.device)
        else:
            # Use a single, unified network
            self.actor_critic = ActorCriticCNN(
                env.single_observation_space.shape, env.single_action_space.n, hidden_size
            ).to(self.device)
            self.target_critic = ActorCriticCNN(
                env.single_observation_space.shape, env.single_action_space.n, hidden_size
            ).to(self.device)

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        # Copy weights
        self.target_critic.load_state_dict(self.actor_critic.state_dict())
        # A placeholder to track the current states in between rollouts
        self.current_states = None

    def soft_update_target_network(self):
        """
        Soft target network update
        """
        target_net_state_dict = self.target_critic.state_dict()
        online_net_state_dict = self.actor_critic.state_dict()
        for key in online_net_state_dict:
            target_net_state_dict[key] = online_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                1 - self.tau
            )
        self.target_critic.load_state_dict(target_net_state_dict)

    def choose_action(self, state: np.ndarray) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """
        Returns N actions per N envs
        """
        logits, _ = self.actor_critic(state)
        probs = F.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        return action_dist.sample()

    def get_action_value(self, state: np.ndarray) -> Tuple[torch.distributions.Categorical, torch.Tensor, torch.Tensor]:
        logits, value = self.actor_critic(state)
        probs = F.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        actions = action_dist.sample()
        return action_dist, actions, value

    def evaluate(self, state: np.ndarray) -> torch.Tensor:
        """
        Returns N V(s) per N envs
        """
        return self.actor_critic.get_value(state)

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
        # Initialize advantages and last gae, shape (N, 1)
        advantages = torch.zeros_like(values).to(self.device)
        last_gae_lambda = torch.zeros(self.num_envs).to(self.device)

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
        states = self.current_states

        # Lists to store the N step data
        values_list, logprobs_list, entropies_list, rewards_list, dones_list = [], [], [], [], []
        for _ in range(self.num_steps):
            # Actions.shape: (N,)
            action_dist, actions, state_values = self.get_action_value(states)
            # Step in the environments
            next_states, rewards, terminateds, _, infos = self.env.step(actions.cpu().numpy())

            # Store episodic rewards from all envs
            # Note if some envs are done, they will be automatically resetted
            # No need to manually call reset()
            if "_episode" in infos:
                # infos["_episode"] is a boolean array, True for envs that finished
                finished_env_indices = np.where(infos["_episode"])[0]
                for env_idx in finished_env_indices:
                    # Access the final reward from the 'episode' dictionary for that specific env
                    episode_reward = infos["episode"]["r"][env_idx]
                    self.train_rewards.append(episode_reward)

            # Store data
            logprobs_list.append(action_dist.log_prob(actions))  # Gradient tracked
            values_list.append(state_values)  # Gradient tracked
            entropies_list.append(action_dist.entropy())  # Gradient tracked
            rewards_list.append(rewards)
            dones_list.append(terminateds)

            states = next_states

        # Convert data that does NOT need gradients
        rewards_t = torch.tensor(np.array(rewards_list), dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(np.array(dones_list), dtype=torch.float32).to(self.device)

        # Stack tensors that DO need gradients.
        # torch.stack is a differentiable operation that preserves the computation graph.
        logprobs_t = torch.stack(logprobs_list)
        values_t = torch.stack(values_list)
        entropies_t = torch.stack(entropies_list)

        # Save current state for the future rollout
        self.current_states = states

        # Find the value of the final state using the target critic
        with torch.no_grad():
            # next_values_t = self.online_critic(states)
            next_values_t = self.evaluate(states)

        # Calculate returns and advantages
        returns_t, advantages_t = self.compute_returns_and_advantages(
            rewards_t, dones_t, values_t.squeeze(), next_values_t.squeeze()
        )

        return logprobs_t, values_t, entropies_t, returns_t, advantages_t

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
        actor_loss = -(logprobs * advantages).mean()

        # Add entropy to the loss, if specified
        if self.use_entropy:
            entropy_loss = self.entropy_coef * entropies.mean()
            actor_loss -= entropy_loss

        # Critic loss: L2 between target and state values
        critic_loss = F.mse_loss(values.squeeze(), returns)
        total_loss = actor_loss + self.critic_scale * critic_loss

        # Zero out the gradients
        self.optimizer.zero_grad()
        total_loss.backward()
        # Clip the gradients for stability
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=1)
        # Step the optimizer
        self.optimizer.step()

    def train(self, max_steps: int = 10000):
        pbar = tqdm(range(max_steps), desc="Training")

        states, _ = self.env.reset()

        # In the beginning set the initial state
        self.current_states = states
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
        save_path = os.path.join(save_path, "actor_critic.pth")
        torch.save(self.actor_critic.state_dict(), save_path)

    def load(self, load_path: str):
        """
        Load pre-trained policy
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Load path does not exist: {load_path}")
        load_path = os.path.join(load_path, "actor_critic.pth")
        self.actor_critic.load_state_dict(torch.load(load_path))
