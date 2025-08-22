import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from core.configs import AgentConfig
from core.DQNBase import DQNBase
from core.policies import NoisyCNN, NoisyDuelingCNN, NoisyDuelingFC, NoisyFC
from tqdm import tqdm


class AgentDQN(DQNBase):
    """
    Noisy Net DQN algorithm with Prioritized Experience Replay, Double DQN update rule, and N-step returns.
    Uses parameter noise for exploration instead of epsilon-greedy.
    """

    def __init__(self, env: gym.Env, agent_config: AgentConfig, solved_threshold: float, is_atari: bool):

        # Overwrite some specific params
        agent_config.memory = "per"
        agent_config.dueling = True
        agent_config.noisy_net = True
        # Initialize the parent class
        super().__init__(env, agent_config, is_atari, solved_threshold)

        self.name = "Noisy-DQN"

        # ---------- PER PARAMS ------------
        self.beta_start = agent_config.beta_start
        self.beta_end = agent_config.beta_final
        self.beta_increase_steps = agent_config.beta_increase_steps
        self.current_beta = self.beta_start
        self.learning_step_count = 0

        # ---------- NOISY NET PARAMS ------------
        self.noisy_std = agent_config.noise_std

        # Override the policy networks with noisy versions
        self._setup_noisy_networks(agent_config)

    def _setup_noisy_networks(self, agent_config: AgentConfig):
        """Setup noisy networks instead of regular Q-networks"""
        if not self.is_atari:
            if agent_config.dueling:
                self.online_model = NoisyDuelingFC(
                    self.observation_space.shape[0], self.action_space.n, agent_config.hidden_dim, self.noisy_std
                ).to(self.device)
                self.target_model = NoisyDuelingFC(
                    self.observation_space.shape[0], self.action_space.n, agent_config.hidden_dim, self.noisy_std
                ).to(self.device)
            else:
                self.online_model = NoisyFC(
                    self.observation_space.shape[0], self.action_space.n, agent_config.hidden_dim, self.noisy_std
                ).to(self.device)
                self.target_model = NoisyFC(
                    self.observation_space.shape[0], self.action_space.n, agent_config.hidden_dim, self.noisy_std
                ).to(self.device)
        else:
            if agent_config.dueling:
                self.online_model = NoisyDuelingCNN(
                    self.observation_space.shape, self.action_space.n, agent_config.hidden_dim, self.noisy_std
                ).to(self.device)
                self.target_model = NoisyDuelingCNN(
                    self.observation_space.shape, self.action_space.n, agent_config.hidden_dim, self.noisy_std
                ).to(self.device)
            else:
                self.online_model = NoisyCNN(
                    self.observation_space.shape, self.action_space.n, agent_config.hidden_dim, self.noisy_std
                ).to(self.device)
                self.target_model = NoisyCNN(
                    self.observation_space.shape, self.action_space.n, agent_config.hidden_dim, self.noisy_std
                ).to(self.device)

        # Copy the initial weights
        self.hard_update_target_network()

        # Re-initialize optimizer with new model parameters
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.lr)

    def choose_action(self, state: np.ndarray, epsilon: Optional[float] = None):
        """
        Greedy policy with noisy networks.
        No epsilon-greedy needed as exploration is handled by parameter noise.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_model(state_tensor)
            action = torch.argmax(q_values, dim=1).squeeze().item()
            return action

    def reset_noise(self):
        """Reset noise in both online and target networks"""
        self.online_model.reset_noise()
        self.target_model.reset_noise()

    def learn(self):
        """
        Implements learning with Noisy Networks and Prioritized Experience Replay (PER).
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        # Anneal beta for PER
        progress = min(self.learning_step_count / self.beta_increase_steps, 1.0)
        self.current_beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        # 1. Sample a batch of prioritized experiences from PER buffer
        states, actions, rewards, next_states, dones, ns, weights, indices = self.memory.sample(
            self.batch_size, self.current_beta
        )

        # 2. Cast np.arrays into torch.Tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        ns = torch.LongTensor(ns).to(self.device)

        # 3. Calculate Q-values Q(s,a)
        q_values = self.online_model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 4. Calculate target Q-values using Double DQN
        with torch.no_grad():
            # Disable noise first
            self.online_model.eval()
            self.target_model.eval()
            # Use online network to select best actions
            next_actions = torch.argmax(self.online_model(next_states), dim=1)

            # Use target network to evaluate the Q-values for selected actions
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()

            # Compute n-step return targets
            gamma_n = self.gamma**ns
            targets = rewards + gamma_n * next_q_values * (1 - dones)
            # Enable it back
            self.online_model.train()
            self.target_model.train()
        # 5. Calculate TD-errors for PER
        td_errors = torch.abs(q_values - targets)

        # 6. Calculate loss with importance sampling weights
        loss = F.mse_loss(q_values, targets, reduction="none")
        weighted_loss = weights * loss
        loss = weighted_loss.mean()

        # 7. Perform Gradient Descent Step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), max_norm=5.0)
        self.optimizer.step()

        # 8. Update priorities in PER
        new_priorities = td_errors.detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        # Increment step counter for beta annealing
        self.learning_step_count += 1

        # 9. Reset noise after all gradient computations are done
        self.reset_noise()

        return loss.item()

    def train(self, max_steps: int = 100000, timeout: float = None):
        obs, _ = self.env.reset()
        if self.is_atari:
            obs = self.auto_fire()

        pbar = tqdm(range(max_steps), desc="Training")

        episode = 0
        start_time = time.time()
        val_mean_reward = -float("inf")
        train_mean_reward = -float("inf")
        for global_step in pbar:
            action = self.choose_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            clipped_reward = self.clip_reward(reward) if self.is_atari else reward
            # We are no more adding each experience tuple to the memory, instead we are adding to the memory:
            # - s_t -- Start state
            # - a_t -- Action that was taken
            # - R_{t+n+1} -- Accumulated reward for N steps
            # - done_{t+n+1} -- Whether or not the s_{t+n+1} was terminal or not
            # - n -- number of actual steps that were taken (because the agent can end the episode in steps less than N)
            # So, add a single experience tuple into a separate buffer for futher post-processing
            self.n_step_buffer.append((obs, action, clipped_reward, next_obs, terminated))

            # If the buffer has enough steps process data and store it into memory
            if len(self.n_step_buffer) == self.n_step_return:
                # Get accumulated reward, final next_state and final done
                n_step_reward, n_step_next_state, n_step_done, n = self._get_n_step_info()
                # Get a start state and action taken in that state (it is the first transition in the n_step_buffer)
                start_state, start_action, _, _, _ = self.n_step_buffer[0]
                # Store it into memory
                self.memory.push(start_state, start_action, n_step_reward, n_step_next_state, n_step_done, n)

            if global_step > self.learning_starts and global_step % self.learning_freq == 0:
                self.learn()
                if self.should_update_target(self.learning_step_count):
                    self.update_target_network()

            if done:
                # Episode finished, flush the n-step buffer
                self.n_step_buffer.clear()

                if "episode" in info:
                    self.train_rewards.append(info["episode"]["r"])
                episode += 1
                train_mean_reward = np.mean(self.train_rewards[-100:])

                # Evaluate
                if global_step > self.learning_starts and episode % self.evaluation_period == 0:
                    # Disable noise for the evaluation
                    self.online_model.eval()
                    val_mean_reward = self.evaluate()
                    # Enable it back
                    self.online_model.train()
                    self.val_rewards.append(val_mean_reward)
                    if val_mean_reward > self.solved_threshold:
                        print(f"Solved! Mean reward: {val_mean_reward}")
                        break

                obs, _ = self.env.reset()
                if self.is_atari:
                    obs = self.auto_fire()
            else:
                obs = next_obs
            # Log out the metrics
            postfix = {"Ep.": episode, "Train": f"{train_mean_reward:.2f}", "Val": f"{val_mean_reward:.2f}"}
            pbar.set_postfix(postfix)

            if timeout is not None and time.time() - start_time > timeout * 60:
                print("[bold red] Timeout has expired, finishing the training...")
                break

        pbar.close()
