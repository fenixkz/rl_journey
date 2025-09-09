import os
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from core.actor_critic import Actor, Critic
from core.memory import ReplayBuffer
from torch.optim import AdamW
from tqdm import tqdm

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../../")
sys.path.append(os.path.abspath(parent_path))

from common.base_agent import BaseAgent  # noqa: E402


class SACAgent(BaseAgent):

    def __init__(
        self,
        env_id: str,
        solved_threshold: float,
        num_envs: int = 8,
        seed: int = 24,
        memory_size: int = int(1e6),
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        batch_size: int = 256,
        tau: float = 0.005,
        evaluation_period: int = 100,
        evaluation_episodes: int = 10,
        init_alpha: float = 0.2,
        target_entropy: float = None,
        auto_entropy_tuning: bool = True,
        actor_update_rate: int = 1,
    ):
        # SAC can easily work with parallel envs
        env_fns = [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(env_id)) for _ in range(num_envs)]
        env = gym.vector.SyncVectorEnv(env_fns)
        assert isinstance(env.single_action_space, gym.spaces.Box), "SAC is only for continuous space"

        # Initialize the base class
        super().__init__(env=env, solved_threshold=solved_threshold, seed=seed)

        # --- Initialize online and target networks ---
        # --- Actor (stochastic policy) ---
        self.online_actor = Actor(
            env.single_observation_space.shape[0],
            env.single_action_space.shape[0],
            env.single_action_space.high,
            env.single_action_space.low,
        ).to(self.device)

        # --- Critic ---
        # SAC uses twin critics similar to TD3, but defined in a single class
        self.online_critic = Critic(env.single_observation_space.shape[0], env.single_action_space.shape[0]).to(
            self.device
        )

        # Target critic only (no target actor in SAC)
        self.target_critic = Critic(env.single_observation_space.shape[0], env.single_action_space.shape[0]).to(
            self.device
        )
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # --- Loss, Optimizer ---
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = AdamW(self.online_actor.parameters(), lr=actor_lr)
        self.critic_optimizer = AdamW(self.online_critic.parameters(), lr=critic_lr)

        # --- Temperature parameter (alpha) for entropy regularization ---
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            # Target entropy is -dim(A)
            if target_entropy is None:
                self.target_entropy = -torch.prod(torch.Tensor(env.single_action_space.shape).to(self.device)).item()
            else:
                self.target_entropy = target_entropy

            # Learnable temperature parameter, requires_grad has to be set to True
            # Same trick to learn a log(alpha) to force it being always positive
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = AdamW([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = init_alpha

        # --- Replay buffer ---
        self.memory = ReplayBuffer(memory_size)

        # --- Hyperparameters ---
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.evaluation_period = evaluation_period
        self.evaluation_episodes = evaluation_episodes
        self.actor_update_rate = actor_update_rate
        self.num_envs = num_envs

        # --- Initialize the boundaries for actions ---
        self.action_low = torch.FloatTensor(env.single_action_space.low).to(self.device)
        self.action_high = torch.FloatTensor(env.single_action_space.high).to(self.device)

        # Counter for global learning steps
        self.global_step = 0

    @torch.no_grad()
    def choose_action(self, states, deterministic: bool = False) -> np.ndarray:
        # The actor now expects a batch of states (num_envs, obs_dim)
        if deterministic:
            # For evaluation, use mean of the distribution
            _, _, actions = self.online_actor.sample(states)
        else:
            # For exploration, sample from the distribution
            actions, _, _ = self.online_actor.sample(states)

        # Actions are already clamped in the actor, move to cpu and cast to np.ndarray
        return actions.cpu().numpy()

    def soft_update(self):
        """Soft update the target critic network with the online critic's weights using tau."""
        for target_param, online_param in zip(self.target_critic.parameters(), self.online_critic.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    def learn(self):
        # First get the data
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        #  Then convert all entities into tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        # Good to be explicit with shapes
        reward = torch.FloatTensor(reward).to(self.device).view(self.batch_size, 1)
        done = torch.LongTensor(done).to(self.device).view(self.batch_size, 1)

        # --- SAC update rule ---
        # --- Critic Update ---
        # 1. Compute TD-target
        with torch.no_grad():
            # 1. Sample next action from the policy (only online)
            next_action, next_log_prob, _ = self.online_actor.sample(next_state)
            # 2. Get Q-values from target twin critics
            target_q1, target_q2 = self.target_critic(next_state, next_action)
            # 3. Calculate soft TD-target using the min among those two, with entropy bonus
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target = reward + self.gamma * target_q * (1 - done)

        # 2. Get Q-values of current states action pairs using twins
        current_q1, current_q2 = self.online_critic(state, action)
        # 3. Calculate MSE loss
        loss_q1 = self.critic_criterion(current_q1, target)
        loss_q2 = self.critic_criterion(current_q2, target)
        critic_loss = loss_q1 + loss_q2
        # 4. Backpropagate
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        # SAC also delays actor update
        if self.global_step % self.actor_update_rate == 0:
            # 1. Sample action from the current policy
            new_action, new_log_prob, _ = self.online_actor.sample(state)
            # 2. Get Q-value from both critics
            q1, q2 = self.online_critic(state, new_action)

            # 3. Maximize Q-value and entropy
            actor_loss = ((self.alpha * new_log_prob) - torch.min(q1, q2)).mean()

            # 4. Backpropagate
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # --- Temperature (alpha) update ---
            if self.auto_entropy_tuning:
                # Since the actor has changed after optimizer step, re-calculate log_prob
                # No gradients must be tracked to actor
                with torch.no_grad():
                    _, new_log_prob, _ = self.online_actor.sample(state)
                # Calculate the loss
                alpha_loss = -(self.log_alpha.exp() * (new_log_prob + self.target_entropy)).mean()
                # Backpropogate
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                # Re-assign alpha
                self.alpha = self.log_alpha.exp().item()

            # Soft update aligns with actor update
            self.soft_update()

        # Increment global learning step
        self.global_step += 1

    def evaluate(self):
        val_rewards = []
        states, _ = self.env.reset()
        while len(val_rewards) < self.evaluation_episodes:
            actions = self.choose_action(states, deterministic=True)
            next_states, _, _, _, infos = self.env.step(actions)
            if "_episode" in infos:
                finished_env_indices = np.where(infos["_episode"])[0]
                for env_idx in finished_env_indices:
                    if len(val_rewards) < self.evaluation_episodes:
                        val_rewards.append(infos["episode"]["r"][env_idx])
            states = next_states

        return np.mean(val_rewards)

    def train(self, max_steps: int = 10000):
        pbar = tqdm(range(max_steps), desc="Training")
        val_avg_reward = -float("inf")
        train_avg_reward = -float("inf")

        states, _ = self.env.reset()
        for step in pbar:
            # Start with random actions to fill buffer
            if step < self.batch_size * 100:
                actions = self.env.action_space.sample()
            else:
                actions = self.choose_action(states)

            next_states, rewards, terminateds, _, infos = self.env.step(actions)
            for idx in range(self.num_envs):
                self.memory.push(states[idx], actions[idx], rewards[idx], next_states[idx], terminateds[idx])

            self.learn()
            states = next_states
            if "_episode" in infos:
                for env_idx in np.where(infos["_episode"])[0]:
                    self.train_rewards.append(infos["episode"]["r"][env_idx])

            if self.train_rewards:
                train_avg_reward = np.mean(self.train_rewards[-100:])

            # Time to evaluate, note: on step basis
            if (step + 1) % self.evaluation_period == 0:
                val_avg_reward = self.evaluate()
                self.val_rewards.append(val_avg_reward)
                if val_avg_reward > self.solved_threshold:
                    pbar.set_description_str("Environment solved!")
                    break

            pbar.set_postfix_str(
                f"Ep. {len(self.train_rewards)} | Train {train_avg_reward:.1f} | \
                Val. {val_avg_reward:.1f} | α {self.alpha:.3f}"
            )

    def save(self, save_path: str):
        """Saves the state dictionaries of the online actor and critic."""
        print("--- Saving models ---")
        torch.save(self.online_actor.state_dict(), os.path.join(save_path, "actor.pth"))
        torch.save(self.online_critic.state_dict(), os.path.join(save_path, "critic.pth"))
        if self.auto_entropy_tuning:
            torch.save(self.log_alpha, os.path.join(save_path, "log_alpha.pth"))

    def load(self, load_path: str):
        """Loads the state dictionaries for the online actor and critic."""
        print("--- Loading models ---")
        actor_w = torch.load(os.path.join(load_path, "actor.pth"))
        critic_w = torch.load(os.path.join(load_path, "critic.pth"))
        self.online_actor.load_state_dict(actor_w)
        self.online_critic.load_state_dict(critic_w)
        if self.auto_entropy_tuning and os.path.exists(os.path.join(load_path, "log_alpha.pth")):
            self.log_alpha = torch.load(os.path.join(load_path, "log_alpha.pth"))
            self.alpha = self.log_alpha.exp()
