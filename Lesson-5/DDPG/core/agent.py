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


class DDPGAgent(BaseAgent):

    def __init__(
        self,
        env_id: str,
        solved_threshold: float,
        seed: int = 24,
        memory_size: int = int(1e6),
        initial_noise: float = 0.1,
        noise_decay: float = 0.9995,
        min_noise: float = 0.01,
        gamma: float = 0.99,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        batch_size: int = 256,
        tau: float = 0.005,
        evaluation_period: int = 100,
        evaluation_episodes: int = 10,
    ):
        env = gym.make(env_id)
        assert isinstance(env.action_space, gym.spaces.Box), "DDPG is only for continuos space"
        # Initialize the base class
        super().__init__(env=env, solved_threshold=solved_threshold, seed=seed)

        # --- Initialize online and target networks ---
        # --- Actor ---
        self.online_actor = Actor(
            env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, env.action_space.low
        ).to(self.device)
        self.target_actor = Actor(
            env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, env.action_space.low
        ).to(self.device)
        # --- Critic ---
        self.online_critic = Critic(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
        self.target_critic = Critic(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)

        # --- Loss, Optimizer ---
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = AdamW(self.online_actor.parameters(), lr=actor_lr)
        self.critic_optimizer = AdamW(self.online_critic.parameters(), lr=critic_lr)

        # --- Copy online weights to target networks ---
        self.target_actor.load_state_dict(self.online_actor.state_dict())
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # --- Replay buffer ---
        self.memory = ReplayBuffer(memory_size)

        # --- Hyperparameters ---
        self.noise_decay = noise_decay
        self.min_noise_magnitude = min_noise
        self.noise_magnitude = initial_noise
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.evaluation_period = evaluation_period
        self.evaluation_episodes = evaluation_episodes

        # --- Initialize the boundaries for actions ---
        self.action_low = torch.FloatTensor(env.action_space.low).to(self.device)
        self.action_high = torch.FloatTensor(env.action_space.high).to(self.device)

    @torch.no_grad()
    def choose_action(self, state, deterministic: bool = False) -> np.ndarray:
        # Get an action from the online actor
        action = self.online_actor(state)
        # Add exploration noise
        # The noise have to be sampled from Gaussian of zero mean
        # And standard deviation of action range (upper-lower)/2
        if not deterministic:
            noise = torch.normal(0, self.online_actor.range * self.noise_magnitude)
            action += noise

        # Clamp the action to the boundaries, move to cpu and remove batch dim
        return torch.clamp(action, self.action_low, self.action_high).cpu().numpy().squeeze(0)

    def decay_noise(self):
        if self.noise_magnitude > self.min_noise_magnitude:
            self.noise_magnitude *= self.noise_decay
        else:
            self.noise_magnitude = self.min_noise_magnitude

    def soft_update(self):
        """Soft update the target networks with the online networks' weights using tau."""
        for target_param, online_param in zip(self.target_critic.parameters(), self.online_critic.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

        for target_param, online_param in zip(self.target_actor.parameters(), self.online_actor.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    def learn(self):
        # Pass if not enough data
        if len(self.memory) < self.batch_size:
            return
        # First get the data
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        #  Then convert all entities into tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        # Good to be explicit with shapes
        reward = torch.FloatTensor(reward).to(self.device).view(self.batch_size, 1)
        done = torch.LongTensor(done).to(self.device).view(self.batch_size, 1)

        # --- DPG update rule ---
        # --- Critic Update ---
        # 1. Compute TD-target
        # All operations done by target network should not be tracked!
        with torch.no_grad():
            # 1. Get the actions for next states using target actor
            # Note, no noise is injected
            next_actions = self.target_actor(next_state)
            # 2. Get Q-values of next-state next-actions using target critic
            next_qs = self.target_critic(next_state, next_actions)
            # 3. Calculate TD-target
            target = reward + self.gamma * next_qs * (1 - done)

        # 2. Get Q-values of current states action pairs using online critic
        q_values = self.online_critic(state, action)
        # 3. Calculate MSE loss
        critic_loss = self.critic_criterion(q_values, target)
        # 4. Backpropogate
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        """
        Order matters here, first the critic has to be updated.
        Because actor's gradients depend on the critic gradients wrt to the action.
        So, we should update critic's weight before updating actor's weights
        """

        # --- Actor Update ---
        # 1. Get the updated action from the given state
        # Note: no noise injected
        new_actions = self.online_actor(state)
        # 2. Ask up-to-date online critic to find its Q-value
        new_q_values = self.online_critic(state, new_actions)
        # 3. Set the actor loss to the negative of this Q-value
        actor_loss = -1 * new_q_values.mean()

        # 4. Backpropogate
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def evaluate(self):
        val_rewards = []
        for _ in range(self.evaluation_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.choose_action(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action=action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
            val_rewards.append(episode_reward)
        return np.mean(val_rewards)

    def train(self, num_episodes: int = 10000):
        pbar = tqdm(range(num_episodes), desc="Training")
        val_avg_reward = -float("inf")

        for e in pbar:
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action=action)
                done = terminated or truncated
                episode_reward += reward
                self.memory.push(state, action, reward, next_state, done)

                self.learn()
                self.soft_update()
                state = next_state

            self.train_rewards.append(episode_reward)
            self.decay_noise()
            train_avg_reward = np.mean(self.train_rewards[-100:])

            if (e + 1) % self.evaluation_period == 0:  # Time to evaluate
                val_avg_reward = self.evaluate()
                self.val_rewards.append(val_avg_reward)
            pbar.set_postfix_str(
                f"Ep. {e} | Train {train_avg_reward:.1f} | Val. {val_avg_reward:.1f} | \
                Noise: {self.noise_magnitude:.3f}"
            )
            if val_avg_reward > self.solved_threshold:
                pbar.set_description_str("Enviroment solved!")
                break

    def save(self, save_path: str):
        """Saves the state dictionaries of the online actor and critic."""
        print("--- Saving models ---")
        torch.save(self.online_actor.state_dict(), os.path.join(save_path, "actor.pth"))
        torch.save(self.online_critic.state_dict(), os.path.join(save_path, "critic.pth"))

    def load(self, load_path: str):
        """Loads the state dictionaries for the online actor and critic."""
        print("--- Loading models ---")
        actor_w = torch.load(os.path.join(load_path, "actor.pth"))
        critic_w = torch.load(os.path.join(load_path, "critic.pth"))
        self.online_actor.load_state_dict(actor_w)
        self.online_critic.load_state_dict(critic_w)
