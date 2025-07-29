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
import torch.nn as nn

class FCD3QN(nn.Module):
    def __init__(self, obs_space, action_space, hidden_space: int = 128):
        super(FCD3QN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_space, hidden_space),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space)
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, 1) # A single output V(s)
        )

    def forward(self, x: torch.Tensor):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
    
class CNND3QN(nn.Module):
    def __init__(self, obs_space, action_space, hidden_space: int = 128):
        super(CNND3QN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(obs_space[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )   
        # Pass a zero tensor to get the final flattened shape of the resulting tensor
        with torch.no_grad():
            feature_size = self.backbone(torch.zeros(1, *obs_space)).view(1, -1).size(1)
        
        self.advantage = nn.Sequential(
            nn.Linear(feature_size, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space)
        )
        
        self.value = nn.Sequential(
            nn.Linear(feature_size, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, 1) # A single output V(s)
        )

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class D3QN(DQNAgent):
    '''
    A Dueling Double DQN algorithm.
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

        # Replace the parent's models with the Dueling architecture
        print(f"Replacing policies to match Dueling-DQN implementation")
        if not self.is_atari:
            print(f"Detected a classic (vector) environment, observation space shape: {self.observation_space.shape}, using Fully Connected (FC) Network")
            hidden_space = kwargs.get("hidden_space", 64)
            self.online_model = FCD3QN(self.observation_space.shape[0], self.action_space.n, hidden_space).to(self.device)
            self.target_model = FCD3QN(self.observation_space.shape[0], self.action_space.n, hidden_space).to(self.device)
        else:
            print(f"Detected an Atari environment, observation space shape: {self.observation_space.shape}, using Convolutional Neural Network (CNN)")
            hidden_space = kwargs.get("hidden_space", 512)
            self.online_model = CNND3QN(self.observation_space.shape, self.action_space.n, hidden_space).to(self.device)
            self.target_model = CNND3QN(self.observation_space.shape, self.action_space.n, hidden_space).to(self.device)

        # Re-nitialize optimizer, use same set of params as in Nature paper
        self.optimizer = torch.optim.RMSprop(self.online_model.parameters(), lr=self.lr, alpha=0.95, eps=0.01, momentum=0.95, centered=False)
        # Sync the networks
        self.hard_update_target_network()

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
            
            # 1. Use online model to find the indexes of the best action in the next states
            online_next_q = self.online_model(next_states)  # Shape: [batch_size, action_space.n]
            next_best_actions = torch.argmax(online_next_q, dim = 1) # Shape: [batch_size]
            
            # 2. Evaluate Q-values for the next states using target network and pick Q-values according to the indexes of next_best_action
            target_new_q: torch.Tensor = self.target_model(next_states) # Shape: [batch_size, action_space.n]
            target_q = target_new_q.gather(1, next_best_actions.unsqueeze(-1)).squeeze(-1) # Shape: [batch_size]

            # 2. Calculate TD target
            # Target = reward + gamma * Q_target(s', argmax(Q_online(s'))) * (1 - done)
            # Multiply by (1 - done) so target is just 'r' if next_state is terminal
            td_target = rewards + self.gamma * target_q * (1 - dones)

        # 6. Calculate loss using Huber Loss (Smooth L1 Loss)
        # This correctly implements the TD-error clipping from the Nature paper.
        loss = F.smooth_l1_loss(actual_q_values, td_target)

        # 7. Perform Gradient Descent Step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), max_norm=2.0)
        self.optimizer.step()

    def train(self, mean_rewards: List, std_rewards: List, max_steps: int = 100000, mean_n_episodes: int = 50):
        rewards_log = deque(maxlen=mean_n_episodes)
        
        obs, _ = self.env.reset(seed=self.seed)
        if self.is_atari: obs = self.auto_fire()

        pbar = tqdm(range(max_steps), desc="Training", postfix={"mean_reward": "N/A", "avg_loss": "N/A"})

        learning_step = 0

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