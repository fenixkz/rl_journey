import os
import sys
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from core.actor_critic import ActorCriticCNN, ActorCriticFC
from core.rollout_buffer import RolloutBuffer
from tqdm import tqdm

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../../../")
sys.path.append(os.path.abspath(parent_path))

from common.base_agent import BaseAgent  # noqa: E402
from common.utils.atari_utils import get_atari_env  # noqa: E402


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) with Generalized Advantage Estimate (GAE)
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
        num_mini_batches: int = 4,
        clip_epsilon: float = 0.2,
        update_epochs: int = 5,
        anneal_lr: bool = False,
        clip_values: bool = False,
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
        self.use_normalization = use_normalization
        self.use_entropy = use_entropy
        self.entropy_coef = entropy_coef
        self.num_steps = num_steps
        self.tau = tau
        self.critic_scale = critic_scale
        self.num_mb = num_mini_batches
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.clip_values = clip_values

        # --- Rollout buffer ---
        self.rollout_buffer = RolloutBuffer(
            num_steps=num_steps,
            obs_shape=env.single_observation_space.shape,
            num_envs=num_envs,
            device=self.device,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        # --- Policy: Actor, Critic --
        if not is_atari:
            self.actor_critic = ActorCriticFC(
                env.single_observation_space.shape[0], env.single_action_space.n, hidden_size
            ).to(self.device)
            self.target_critic = ActorCriticFC(
                env.single_observation_space.shape[0], env.single_action_space.n, hidden_size
            ).to(self.device)
        else:
            # CNN version uses a shared backbone
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

    def collect_rollout(self):
        """
        Collects `num_steps` of experience from environment.
        """
        # Query the state from the last rollout
        states = self.current_states

        for step in range(self.num_steps):
            # No need for gradients in PPO, because we re-compute them later
            with torch.no_grad():
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
            self.rollout_buffer.add(
                step=step,
                states=states,
                actions=actions,
                logprobs=action_dist.log_prob(actions),
                rewards=rewards,
                dones=terminateds,
                values=state_values.squeeze(),
            )
            # Transit
            states = next_states

        # Save current state for the future rollout
        self.current_states = states
        # Find the value of the final state (M+1) using the target critic
        with torch.no_grad():
            next_values = self.evaluate(states)
        return next_values.squeeze()

    def learn(self):
        # Run a single batch several epochs
        for _ in range(self.update_epochs):
            # Now do mini-batch gradient descent
            for batch in self.rollout_buffer.get_mini_batches(self.num_mb):
                # Access data directly from the yielded dictionary
                mb_states = batch["states"]
                mb_actions = batch["actions"]
                mb_logprobs = batch["logprobs"]
                mb_advantages = batch["advantages"]
                mb_returns = batch["returns"]
                mb_values = batch["values"]

                # Re-compute logprobs, entropies for states and actions we collected
                # Additionally, new state-value is computed
                new_action_dist, _, new_state_values = self.get_action_value(mb_states)
                new_logprobs = new_action_dist.log_prob(mb_actions)
                new_entropies = new_action_dist.entropy()

                # Use logarithm property: r_t = pi_new / pi_old
                # -> log(r_t) = log(pi_new) - log(pi_old)
                logratio = new_logprobs - mb_logprobs
                # r_t = e^(log(r_t))
                ratio = logratio.exp()
                if self.use_normalization:  # Note: we do normalization on mini-batch level
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # --- Actor Loss ---
                unclamped_actor_loss = -mb_advantages * ratio
                clamped_actor_loss = -mb_advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                # Use max instead of min because of flipped sign
                actor_loss = torch.max(unclamped_actor_loss, clamped_actor_loss).mean()
                if self.use_entropy:
                    actor_loss -= self.entropy_coef * new_entropies.mean()

                # --- Critic Loss ---
                # Squeeze last single dim
                new_state_values = new_state_values.squeeze(-1)
                # Standard MSE loss
                if self.clip_values:
                    critic_loss_unclipped = F.mse_loss(new_state_values, mb_returns)
                    value_clipped = mb_values + torch.clamp(
                        new_state_values - mb_values, -self.clip_epsilon, self.clip_epsilon
                    )
                    critic_loss_clipped = F.mse_loss(value_clipped, mb_returns)
                    critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped)
                else:
                    critic_loss = F.mse_loss(new_state_values, mb_returns)

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
            # Collect a rollout of N steps in M envs
            next_values = self.collect_rollout()
            # Calculate returns and advantages
            # They are stored inside the buffer
            self.rollout_buffer.compute_returns_and_advantages(next_values)
            # PPO update rule
            self.learn()
            # Target soft update
            self.soft_update_target_network()
            # Reset the rollout buffer
            self.rollout_buffer.reset()
            # Check if we solved the env
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
