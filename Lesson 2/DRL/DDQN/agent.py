import os
from base.DQNBase import DQNAgent
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from typing import List
from collections import deque

class DDQN(DQNAgent):

    def __init__(self,
                 env: gym.Env,
                 hidden_space: int = 128,
                 gamma: float = 0.99,
                 epsilon: float = 1,
                 epsilon_decay: float = 0.9995,
                 min_epsilon: float = 0.05,
                 device: str = "cpu",
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 seed: int = 24,
                 lr: float = 3e-4,
                 target_update_freq: int = 500,
                 learning_freq: int = 4,
                 learning_starts: int = 50000,
                 ):
        
        super().__init__(env, hidden_space, gamma, epsilon, epsilon_decay, min_epsilon, device, buffer_size, batch_size, seed, learning_freq)
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=lr)
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts

    def learn(self):
        # 1. Sample a batch of experience from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 2. Cast np.arrays into torch.Tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 3. Calculate Q(s,a) for all states in the batch and all possible actions. Shape: [batch_size, action_space.n]
        q_values: torch.Tensor = self.online_model(states)
        
        # 4. Use gather to select the Q-value for the specific actions in the batch
        actual_q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1) # Shape: [batch_size, 1]
        
        # 5. Calculate TD-target
        with torch.no_grad(): # Target calculations should not affect gradients, this is our target
            # --- DDQN Update Rule ---
            
            # 1. Use online model to find the indeces of the best action in the next states
            online_next_q = self.online_model(next_states)  # Shape: [batch_size, action_space.n]
            next_best_actions = torch.argmax(online_next_q, dim = 1) # Shape: [batch_size]
            
            # 2. Evaluate Q-values for the next states using target network and pick Q-values according to the indexes of next_best_action
            target_new_q: torch.Tensor = self.target_model(next_states) # Shape: [batch_size, action_space.n]
            target_q = target_new_q.gather(1, next_best_actions.unsqueeze(-1)).squeeze(-1) # Shape: [batch_size]

            # 2. Calculate TD target
            # Target = reward + gamma * Q_target(s', argmax(Q_online(s'))) * (1 - done)
            # Multiply by (1 - done) so target is just 'r' if next_state is terminal
            td_target = rewards + self.gamma * target_q * (1 - dones)
        
        # 6. Calculate MSE loss
        loss: torch.Tensor = (td_target - actual_q_values) ** 2
        loss = loss.mean()

        # 7. Perform Gradient Descent Step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def train(self, mean_rewards: List, std_rewards: List, max_steps: int = 100000, mean_n_episodes: int = 50):
        
        # To track performance of training rewards
        rewards_log = deque(maxlen=mean_n_episodes)
        
        obs, _ = self.env.reset()
        
        pbar = tqdm(range(max_steps), desc="Training", postfix={"mean_reward": "N/A"})

        for global_step in pbar:
            # Sample action or choose the best
            action = self.choose_action(obs)
            
            # Decay epsilon based on steps
            self.decay_epsilon()
            
            # Transit in the env
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store experience in the buffer
            self.memory.push(obs, action, reward, next_obs, done)
            
            # Set next observation to the current one
            obs = next_obs
            
            if done: # If done, reset the episode
                # Use the info dict from RecordEpisodeStatistics
                if "episode" in info:
                    rewards_log.append(info['episode']['r'])
                obs, _ = self.env.reset()

            # Only start learning after a certain number of steps have been collected
            if global_step > self.learning_starts:
                # To speed up training for Atari, do learning not at every step, but every L steps
                if global_step % self.learning_freq == 0: self.learn()
                # Update target network every `target_update_freq` steps
                if global_step % self.target_update_freq == 0: self.update_target_network()

            if len(rewards_log) > 0: pbar.set_postfix(mean_reward=f"{np.mean(rewards_log):.2f}", eps=f"{self.epsilon:.3f}")
            
            if len(rewards_log) == mean_n_episodes:
                mean_rewards.append(np.mean(rewards_log))
                std_rewards.append(np.std(rewards_log))