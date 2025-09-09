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


def check_models_are_equal(model1: nn.Module, model2: nn.Module) -> bool:
    """
    Checks if two PyTorch models have the exact same parameters.
    Returns True if they are equal, False otherwise.
    """
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


class TD3Agent(BaseAgent):

    def __init__(
        self,
        env_id: str,
        solved_threshold: float,
        num_envs: int = 8,
        seed: int = 24,
        memory_size: int = int(1e6),
        exploration_noise: float = 0.1,
        gamma: float = 0.99,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        batch_size: int = 256,
        tau: float = 0.005,
        evaluation_period: int = 100,
        evaluation_episodes: int = 10,
        policy_noise: float = 0.2,
        policy_noise_bound: float = 0.5,
        actor_update_rate: int = 2,
    ):
        # TD3 can easily work with parallel envs
        env_fns = [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(env_id)) for _ in range(num_envs)]
        env = gym.vector.SyncVectorEnv(env_fns)
        assert isinstance(env.single_action_space, gym.spaces.Box), "TD3 is only for continuos space"

        # Initialize the base class
        super().__init__(env=env, solved_threshold=solved_threshold, seed=seed)

        # --- Initialize online and target networks ---
        # --- Actor ---
        self.online_actor = Actor(
            env.single_observation_space.shape[0],
            env.single_action_space.shape[0],
            env.single_action_space.high,
            env.single_action_space.low,
        ).to(self.device)
        self.target_actor = Actor(
            env.single_observation_space.shape[0],
            env.single_action_space.shape[0],
            env.single_action_space.high,
            env.single_action_space.low,
        ).to(self.device)
        self.target_actor.load_state_dict(self.online_actor.state_dict())
        # --- Critic ---
        # TD3 novelty, two critics: q1 and q2
        self.online_q1 = Critic(env.single_observation_space.shape[0], env.single_action_space.shape[0]).to(self.device)
        self.online_q2 = Critic(env.single_observation_space.shape[0], env.single_action_space.shape[0]).to(self.device)
        are_critics_equal = check_models_are_equal(self.online_q1, self.online_q2)
        print(f"Online Q1 and Online Q2 are equal: {are_critics_equal}")

        self.target_q1 = Critic(env.single_observation_space.shape[0], env.single_action_space.shape[0]).to(self.device)
        self.target_q1.load_state_dict(self.online_q1.state_dict())
        self.target_q2 = Critic(env.single_observation_space.shape[0], env.single_action_space.shape[0]).to(self.device)
        self.target_q2.load_state_dict(self.online_q2.state_dict())

        # --- Loss, Optimizer ---
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = AdamW(self.online_actor.parameters(), lr=actor_lr)
        self.critic_optimizer = AdamW(
            list(self.online_q1.parameters()) + list(self.online_q2.parameters()), lr=critic_lr
        )

        # --- Replay buffer ---
        self.memory = ReplayBuffer(memory_size)

        # --- Hyperparameters ---
        self.exploration_noise = exploration_noise
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.evaluation_period = evaluation_period
        self.evaluation_episodes = evaluation_episodes
        self.policy_noise = policy_noise
        self.policy_noise_bound = policy_noise_bound
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
        actions = self.online_actor(states)
        if not deterministic:
            # Noise is constant, not decayed anymore
            # This is just easier when we have to deal with parallel envs
            # On top of that it provides constant exploration
            noise = torch.normal(0, self.online_actor.range * self.exploration_noise)
            actions += noise

        # Clamp the actions to the boundaries, move to cpu and cast to np.ndarray
        return torch.clamp(actions, self.action_low, self.action_high).cpu().numpy()

    def soft_update(self):
        """Soft update the target networks with the online networks' weights using tau."""

        def helper(target_tensor: nn.Module, source_tensor: nn.Module):
            for target_param, online_param in zip(target_tensor.parameters(), source_tensor.parameters()):
                target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

        helper(self.target_actor, self.online_actor)
        helper(self.target_q1, self.online_q1)
        helper(self.target_q2, self.online_q2)

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

        # --- TD3 update rule ---
        # --- Critic Update ---
        # 1. Compute TD-target
        with torch.no_grad():
            # 1. TD3 novelty: To get a smoother landscape, inject noise
            noise = (torch.randn_like(action, device=self.device) * self.policy_noise).clamp(
                -self.policy_noise_bound, self.policy_noise_bound
            ) * self.target_actor.range
            # 2. Get the actions for next states using target actor
            # Note, the injected noise is different from the exploration noise
            next_actions = self.target_actor(next_state) + noise
            # Since the noise is added, we have to make sure it is within the action range
            next_actions = torch.clamp(next_actions, self.action_low, self.action_high)
            # 3. TD3 novelty: Get Q-values from twins critics
            next_q1 = self.target_q1(next_state, next_actions)
            next_q2 = self.target_q2(next_state, next_actions)
            # 3. TD3 novelty: Calculate TD-target using the min among those two
            target = reward + self.gamma * torch.min(next_q1, next_q2) * (1 - done)

        # 2. Get Q-values of current states action pairs using twins
        current_q1 = self.online_q1(state, action)
        current_q2 = self.online_q2(state, action)
        # 3. Calculate MSE loss
        loss_q1 = self.critic_criterion(current_q1, target)
        loss_q2 = self.critic_criterion(current_q2, target)
        critic_loss = loss_q1 + loss_q2
        # 4. Backpropogate
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        # TD3 novelty: Actor is updated less frequently
        if self.global_step % self.actor_update_rate == 0:

            # 1. Get the updated action from the given state
            # Note: no noise injected
            new_actions = self.online_actor(state)
            # 2. Ask up-to-date online critic to evaluate new pair
            # TD3 novelty: only the first online critic
            new_q_values = self.online_q1(state, new_actions)
            # 3. Set the actor loss to the negative of this Q-value
            actor_loss = -1 * new_q_values.mean()

            # 4. Backpropogate
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update should happen after actor update
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
                    pbar.set_description_str("Enviroment solved!")
                    break
            pbar.set_postfix_str(
                f"Ep. {len(self.train_rewards)} | Train {train_avg_reward:.1f} | Val. {val_avg_reward:.1f}"
            )

    def save(self, save_path: str):
        """Saves the state dictionaries of the online actor and critic."""
        print("--- Saving models ---")
        torch.save(self.online_actor.state_dict(), os.path.join(save_path, "actor.pth"))
        torch.save(self.online_q1.state_dict(), os.path.join(save_path, "critic.pth"))

    def load(self, load_path: str):
        """Loads the state dictionaries for the online actor and critic."""
        print("--- Loading models ---")
        actor_w = torch.load(os.path.join(load_path, "actor.pth"))
        critic_w = torch.load(os.path.join(load_path, "critic.pth"))
        self.online_actor.load_state_dict(actor_w)
        self.online_q1.load_state_dict(critic_w)
