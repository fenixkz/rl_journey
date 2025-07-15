import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from base.DQNBase import DQNAgent
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from typing import List
from collections import deque

class DQN(DQNAgent):

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
                 update_period: int = 500,
                 ):
        
        super().__init__(env, hidden_space, gamma, epsilon, epsilon_decay, min_epsilon, device, buffer_size, batch_size, seed)
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=lr)
        self.update_period = update_period

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
            # --- DQN Update Rule ---
            
            # 1. Use target model to find the maximum Q-value of all next states in the batch
            next_q_values = self.target_model(next_states)  # Shape: [batch_size, action_space.n]
            next_max_q = torch.max(next_q_values, dim=1).values # Shape: [batch_size, 1]

            # 2. Calculate TD target
            # Target = reward + gamma * max_a Q_target(s', a) * (1 - done)
            # Multiply by (1 - done) so target is just 'r' if next_state is terminal
            td_target = rewards + self.gamma * next_max_q * (1 - dones)
        
        # 6. Calculate MSE loss
        loss: torch.Tensor = (td_target - actual_q_values) ** 2
        loss = loss.mean()

        # 7. Perform Gradient Descent Step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def train(self, mean_rewards: List, std_rewards: List, max_episodes: int = 100000, mean_n_episodes: int = 50):
        # To track performance of last N episodes
        last_n_rewards = deque(maxlen=mean_n_episodes)

        # TQDM for pretty output
        pbar = tqdm(range(max_episodes), desc="Training", postfix={"mean_reward": 0, "eps": 1})

        training_steps = 0
        for e in pbar:
            state, _ = self.env.reset(seed=self.seed)
            done = False
            total_reward = 0

            while not done:
                # Choose action
                action = self.choose_action(state=state)
                next_state, reward, terminated, truncated, _ = self.env.step(action=action)
                done = terminated or truncated
                total_reward += reward

                self.memory.push(state, action, reward, next_state, done)

                if len(self.memory) > self.batch_size:
                    self.learn()
                    # Update target network every M learning steps
                    if (training_steps+1) % self.update_period == 0: self.update_target_network()
                    training_steps += 1

                state = next_state
                
            # Decay epsilon
            self.decay_epsilon()
            last_n_rewards.append(total_reward)
            if len(last_n_rewards) == mean_n_episodes:
                mean_rewards.append(np.mean(last_n_rewards))
                std_rewards.append(np.std(last_n_rewards))
                pbar.set_postfix(mean_reward=f"{np.mean(last_n_rewards):.2f}", eps = f"{self.epsilon:.3f}")
            