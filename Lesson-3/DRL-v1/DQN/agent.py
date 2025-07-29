import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from base.DQNBase import DQNAgent
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from typing import List
from collections import deque

class DQN(DQNAgent):
    '''
    A vanilla DQN algorithm.
    '''
    def __init__(self, *args, **kwargs ):
        # It's better practice to call the parent constructor first.
        # We must pop the D3QN-specific kwargs first, because the parent's
        # __init__ method doesn't accept them and would raise a TypeError.
        target_update_freq = kwargs.pop("target_update_freq", 1000)
        learning_starts = kwargs.pop("learning_starts", 50000)
        learning_freq = kwargs.pop("learning_freq", 1)
        solved_reward = kwargs.pop("solved_reward", 10000)

        super().__init__(*args, **kwargs)
        
        # Now, set the attributes specific to this agent
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.solved_reward = solved_reward

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
            
        # 6. Calculate loss using Huber Loss (Smooth L1 Loss)
        # This correctly implements the TD-error clipping from the Nature paper.
        loss = F.smooth_l1_loss(actual_q_values, td_target)

        # 7. Perform Gradient Descent Step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), max_norm=2.0)
        self.optimizer.step()
        return loss.item()
    
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