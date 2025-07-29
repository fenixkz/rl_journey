import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from base.DQNBase import DQNAgent
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from DDQNPER.PER import PrioritizedReplayBuffer
from typing import List
from collections import deque

class DDQNPER(DQNAgent):
    '''
    A double DQN algorithm with Proritized Experience Replay (PER).
    '''
    def __init__(self, *args, **kwargs ):
        # It's better practice to call the parent constructor first.
        # We must pop the DDQNPER-specific kwargs first, because the parent's
        # __init__ method doesn't accept them and would raise a TypeError.
        target_update_freq = kwargs.pop("target_update_freq", 1000)
        learning_starts = kwargs.pop("learning_starts", 50000)
        learning_freq = kwargs.pop("learning_freq", 1)
        solved_reward = kwargs.pop("solved_reward", 10000)
        
        # PER-specific hyperparameters
        alpha = kwargs.pop("alpha", 0.6)  # How much prioritization is used (0=uniform, 1=full prioritization)
        beta_start = kwargs.pop("beta_start", 0.4)  # Initial importance sampling correction
        beta_end = kwargs.pop("beta_end", 1.0)  # Final importance sampling correction
        # Extract buffer_size but keep it in kwargs for parent constructor
        buffer_size = kwargs.get("buffer_size", 100000)

        super().__init__(*args, **kwargs)
        
        # Now, set the attributes specific to this agent
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.solved_reward = solved_reward
        self.alpha = alpha
        self.current_beta = beta_start  # Current beta value for importance sampling
        self.beta_start = beta_start
        self.beta_end = beta_end
        # Override memory with PER instead of regular ReplayBuffer
        self.memory = PrioritizedReplayBuffer(size=int(buffer_size), alpha=alpha)
        
        # Step counter for beta annealing
        self.step_count = 0

    def learn(self):
        """
        Implements learning with Prioritized Experience Replay (PER).
        
        PER improves sample efficiency by preferentially sampling experiences
        with higher TD-errors (more surprising/informative experiences).
        
        Key differences from uniform sampling:
        1. Sample experiences based on priority (TD-error magnitude)
        2. Use importance sampling weights to correct for sampling bias
        3. Update priorities after computing new TD-errors
        """
        
        # Anneal beta (importance sampling correction) from beta_start to beta_end over training
        # Beta controls how much we correct for the bias introduced by prioritized sampling
        progress = min(self.step_count / 100000, 1.0)  # Normalize step count
        self.current_beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        
        # 1. Sample a batch of prioritized experiences from PER buffer
        # Unlike uniform sampling, this returns additional weights and indices
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size, self.current_beta)
        
        # 2. Cast np.arrays into torch.Tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)  # Importance sampling weights
        
        # 3. Calculate Q(s,a) for all states in the batch and all possible actions
        q_values: torch.Tensor = self.online_model(states)
        
        # 4. Use gather to select the Q-value for the specific actions in the batch
        actual_q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # 5. Calculate TD-target using Double DQN
        with torch.no_grad():
            # --- DDQN Update Rule ---
            # Use online network to select actions, target network to evaluate them
            # This reduces overestimation bias compared to vanilla DQN
            
            # 1. Use online model to find the indexes of the best action in the next states
            online_next_q = self.online_model(next_states)  # Shape: [batch_size, action_space.n]
            next_best_actions = torch.argmax(online_next_q, dim=1)  # Shape: [batch_size]
            
            # 2. Evaluate Q-values for the next states using target network
            target_new_q: torch.Tensor = self.target_model(next_states)  # Shape: [batch_size, action_space.n]
            target_q = target_new_q.gather(1, next_best_actions.unsqueeze(-1)).squeeze(-1)  # Shape: [batch_size]

            # 3. Calculate TD target
            # Target = reward + gamma * Q_target(s', argmax(Q_online(s'))) * (1 - done)
            # Multiply by (1 - done) so target is just 'r' if next_state is terminal
            td_target = rewards + self.gamma * target_q * (1 - dones)

        # 6. Calculate TD-errors for updating priorities
        # TD-error measures how "surprising" or informative each experience is
        td_errors = torch.abs(actual_q_values - td_target)
        
        # 7. Apply importance sampling weights to correct for prioritized sampling bias
        weighted_loss = weights * F.smooth_l1_loss(actual_q_values, td_target, reduction='none')
        loss = weighted_loss.mean()

        # 8. Perform Gradient Descent Step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), max_norm=2.0)
        self.optimizer.step()
        
        # 9. Update priorities in the PER based on new TD-errors
        # Higher TD-error → higher priority → more likely to be sampled in future
        # Add small epsilon to prevent zero priorities
        new_priorities = td_errors.detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, new_priorities)
        
        # Increment step counter for beta annealing
        self.step_count += 1

    def train(self, mean_rewards: List, std_rewards: List, max_steps: int = 100000, mean_n_episodes: int = 50):
        rewards_log = deque(maxlen=mean_n_episodes)
        
        obs, _ = self.env.reset(seed=self.seed)
        if self.is_atari: obs = self.auto_fire()

        pbar = tqdm(range(max_steps), desc="Training", postfix={"mean_reward": "N/A", "avg_loss": "N/A"})

        for global_step in pbar:
            action = self.choose_action(obs)
            self.decay_epsilon()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            clipped_reward = self.clip_reward(reward) if self.is_atari else reward
            # Store experience in PER buffer
            self.memory.push(obs, action, clipped_reward, next_obs, terminated)
            
            if global_step > self.learning_starts and global_step % self.learning_freq == 0:
                self.learn()
                self.update_target_network()   

            if done:
                if "episode" in info: rewards_log.append(info['episode']['r'])
                mean_reward = np.mean(rewards_log)
                if len(rewards_log) == mean_n_episodes:
                    mean_rewards.append(mean_reward)
                    std_rewards.append(np.std(rewards_log))
                if mean_reward > self.solved_reward:
                    print(f"Solved! Mean reward: {mean_reward}")
                    break
                
                obs, _ = self.env.reset()
                if self.is_atari: obs = self.auto_fire()             
            else:
                obs = next_obs
            # Log out the metrics
            postfix = {"mean_reward": f"{mean_reward:.2f}" if rewards_log else "N/A", "eps": f"{self.epsilon:.3f}"}
            pbar.set_postfix(postfix)
        
        pbar.close()