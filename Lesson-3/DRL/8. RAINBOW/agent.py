import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from core.configs import AgentConfig
from core.DQNBase import DQNBase
from core.policies import (
    DistributionalDuelingCNN,
    DistributionalDuelingFC,
    NoisyDistributionalDuelingCNN,
    NoisyDistributionalDuelingFC,
)
from tqdm import tqdm


class AgentDQN(DQNBase):
    """
    RAINBOW: Combining all DQN improvements
    - Double DQN
    - Prioritized Experience Replay (PER)
    - Dueling Architecture
    - Multi-step Returns
    - Distributional DQN (C51)
    - Noisy Networks
    """

    def __init__(self, env: gym.Env, agent_config: AgentConfig, solved_threshold: float, is_atari: bool):

        # Overwrite some specific params
        agent_config.memory = "per"
        agent_config.dueling = True

        # Initialize the parent class
        super().__init__(env, agent_config, is_atari, solved_threshold)

        self.name = "RAINBOW"

        # ---------- PER PARAMS ------------
        self.beta_start = agent_config.beta_start
        self.beta_end = agent_config.beta_final
        self.beta_increase_steps = agent_config.beta_increase_steps
        self.current_beta = self.beta_start
        self.learning_step_count = 0

        # ---------- C51 PARAMS ------------
        self.n_atoms = agent_config.n_atoms
        self.v_min = agent_config.v_min
        self.v_max = agent_config.v_max
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        # Support of the distribution
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)

        # ---------- NOISY NET PARAMS ------------
        self.use_noisy_net = agent_config.noisy_net
        self.noisy_std = agent_config.noise_std

        # Override the policy networks with RAINBOW versions (Noisy + Distributional + Dueling)
        if self.use_noisy_net:
            self._setup_noisy_rainbow_networks(agent_config)
        else:
            self._setup_default_rainbow_networks(agent_config)

    def _setup_default_rainbow_networks(self, agent_config: AgentConfig):
        """Setup RAINBOW networks combining Distributional, and Dueling architectures"""
        if not self.is_atari:
            self.online_model = DistributionalDuelingFC(
                self.observation_space.shape[0], self.action_space.n, agent_config.hidden_dim, self.n_atoms
            ).to(self.device)
            self.target_model = DistributionalDuelingFC(
                self.observation_space.shape[0], self.action_space.n, agent_config.hidden_dim, self.n_atoms
            ).to(self.device)
        else:
            self.online_model = DistributionalDuelingCNN(
                self.observation_space.shape, self.action_space.n, agent_config.hidden_dim, self.n_atoms
            ).to(self.device)
            self.target_model = DistributionalDuelingCNN(
                self.observation_space.shape, self.action_space.n, agent_config.hidden_dim, self.n_atoms
            ).to(self.device)

        # Copy the initial weights
        self.hard_update_target_network()

        # Re-initialize optimizer with new model parameters, Adam is said to be better than RMSProp in RAINBOW paper
        self.optimizer = torch.optim.Adam(
            self.online_model.parameters(), lr=self.lr, eps=1.5e-4
        )  # Per RAINBOW implementation

    def _setup_noisy_rainbow_networks(self, agent_config: AgentConfig):
        """Setup RAINBOW networks combining Noisy, Distributional, and Dueling architectures"""
        if not self.is_atari:
            self.online_model = NoisyDistributionalDuelingFC(
                self.observation_space.shape[0],
                self.action_space.n,
                agent_config.hidden_dim,
                self.n_atoms,
                self.noisy_std,
            ).to(self.device)
            self.target_model = NoisyDistributionalDuelingFC(
                self.observation_space.shape[0],
                self.action_space.n,
                agent_config.hidden_dim,
                self.n_atoms,
                self.noisy_std,
            ).to(self.device)
        else:
            self.online_model = NoisyDistributionalDuelingCNN(
                self.observation_space.shape, self.action_space.n, agent_config.hidden_dim, self.n_atoms, self.noisy_std
            ).to(self.device)
            self.target_model = NoisyDistributionalDuelingCNN(
                self.observation_space.shape, self.action_space.n, agent_config.hidden_dim, self.n_atoms, self.noisy_std
            ).to(self.device)

        # Copy the initial weights
        self.hard_update_target_network()

        # Re-initialize optimizer with new model parameters
        self.optimizer = torch.optim.Adam(
            self.online_model.parameters(), lr=self.lr, eps=1.5e-4
        )  # Per RAINBOW implementation

    def choose_action(self, state: np.ndarray, epsilon: Optional[float] = None):
        """
        Greedy policy with noisy networks.
        No epsilon-greedy needed as exploration is handled by parameter noise.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            log_probs = self.online_model(state_tensor)  # [1, action_space, n_atoms]
            probs = log_probs.exp()

            # Compute expected Q-values: Q(s,a) = sum_i z_i * p_i(s,a)
            q_values = (probs * self.support.view(1, 1, -1)).sum(dim=-1)
            action = torch.argmax(q_values, dim=1).squeeze().item()
            return action

    def reset_noise(self):
        """Reset noise in both online and target networks"""
        self.online_model.reset_noise()
        self.target_model.reset_noise()

    def _project_distribution(self, next_distributions, rewards, dones, ns):
        """
        Project the target distribution onto the support.

        Args:
            next_distributions: Log probabilities of next state distributions [batch_size, n_atoms]
            rewards: Rewards [batch_size]
            dones: Done flags [batch_size]
            ns: Number of steps for n-step return [batch_size]

        Returns:
            Projected distribution [batch_size, n_atoms]
        """
        batch_size = rewards.shape[0]
        delta_z = self.delta_z
        v_min = self.v_min
        v_max = self.v_max
        n_atoms = self.n_atoms
        support = self.support

        # Convert log probabilities to probabilities
        next_probs = next_distributions.exp()

        # Compute the projected support: r + gamma^n * z_j
        # Handle n-step returns properly
        gamma_n = self.gamma ** ns.unsqueeze(1)  # [batch_size, 1]
        rewards = rewards.unsqueeze(1)  # [batch_size, 1]
        dones = dones.unsqueeze(1)  # [batch_size, 1]

        # Projected support: Tz_j = r + gamma^n * z_j * (1 - done)
        Tz = rewards + gamma_n * support.unsqueeze(0) * (1 - dones)
        Tz = Tz.clamp(min=v_min, max=v_max)

        # Compute projection indices
        b = (Tz - v_min) / delta_z
        lower = b.floor().long()
        upper = b.ceil().long()

        # Handle the case where l == u (when b is an integer)
        lower[(upper > 0) * (lower == upper)] -= 1
        upper[(lower < (n_atoms - 1)) * (lower == upper)] += 1

        # Initialize projected distribution
        projected_dist = torch.zeros((batch_size, n_atoms), device=self.device)

        # Distribute probability mass
        offset = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(batch_size, n_atoms)

        # Add probability mass to lower bound
        projected_dist.view(-1).index_add_(
            0, (lower + offset * n_atoms).view(-1), (next_probs * (upper.float() - b)).view(-1)
        )

        # Add probability mass to upper bound
        projected_dist.view(-1).index_add_(
            0, (upper + offset * n_atoms).view(-1), (next_probs * (b - lower.float())).view(-1)
        )

        return projected_dist

    def learn(self):
        """
        Implements learning with RAINBOW, combining all techniques.
        """
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

        # 3. Get current state-action distribution
        current_dist_log = self.online_model(states)  # [batch_size, action_space, n_atoms]
        current_dist_log = current_dist_log.gather(
            1, actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms)
        ).squeeze(1)

        # 4. Calculate target distribution using Double DQN
        with torch.no_grad():
            # Disable noise first
            # self.online_model.eval()
            # self.target_model.eval()
            # Use online network to select best actions
            online_next_log_probs = self.online_model(next_states)
            online_next_probs = online_next_log_probs.exp()
            online_next_q = (online_next_probs * self.support.view(1, 1, -1)).sum(dim=-1)
            next_best_actions = torch.argmax(online_next_q, dim=1)

            # Use target network to evaluate the distribution for selected actions
            target_next_log_probs = self.target_model(next_states)
            target_next_dist = target_next_log_probs.gather(
                1, next_best_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms)
            ).squeeze(1)

            # Project the target distribution
            projected_dist = self._project_distribution(target_next_dist, rewards, dones, ns)

            # Add small epsilon to avoid log(0) in the loss calculation
            projected_dist = projected_dist.clamp(min=1e-8)
            # Enable it back
            # self.online_model.train()
            # self.target_model.train()

        # 5. Calculate cross-entropy loss, current dist_log is already using logartihm
        loss = -(projected_dist * current_dist_log).sum(dim=-1)

        # 6. Store cross-entropy loss for PER priorities (detach to avoid gradients)
        ce_errors = loss.detach()

        # 7. Apply importance sampling weights
        weighted_loss = weights * loss
        loss = weighted_loss.mean()

        # 8. Perform Gradient Descent Step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), max_norm=5.0)
        self.optimizer.step()

        # 9. Update priorities in PER
        new_priorities = ce_errors.cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        # 10. Reset noise after gradient computation
        if self.use_noisy_net:
            self.reset_noise()

        # Increment step counter for beta annealing
        self.learning_step_count += 1

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
                while self.n_step_buffer:
                    n_step_reward, n_step_next_state, n_step_done, n = self._get_n_step_info()
                    start_state, start_action, _, _, _ = self.n_step_buffer.popleft()
                    self.memory.push(start_state, start_action, n_step_reward, n_step_next_state, n_step_done, n)

                if "episode" in info:
                    self.train_rewards.append(info["episode"]["r"])
                episode += 1
                train_mean_reward = np.mean(self.train_rewards[-100:])

                # Evaluate
                if episode % self.evaluation_period == 0:
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
            postfix = {
                "Ep.": episode,
                "Train": f"{train_mean_reward:.2f}",
                "Val": f"{val_mean_reward:.2f}",
            }
            if not self.use_noisy_net:
                postfix["Eps"] = f"{self.epsilon:.3f}"
            pbar.set_postfix(postfix)

            if timeout is not None and time.time() - start_time > timeout * 60:
                print("[bold red] Timeout has expired, finishing the training...")
                break

        pbar.close()
