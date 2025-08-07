import os
import sys
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from typing import List, Dict
from collections import deque
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from core.DQNBase import DQNBase
from core.configs import AgentConfig
from core.policies import NoisyFC, NoisyCNN, NoisyDuelingFC, NoisyDuelingCNN

class AgentDQN(DQNBase):
    '''
    Noisy Net DQN algorithm with Prioritized Experience Replay, Double DQN update rule, and N-step returns.
    Uses parameter noise for exploration instead of epsilon-greedy.
    '''
    def __init__(self, env: gym.Env, agent_config: AgentConfig, env_config: Dict, is_atari: bool):

        # Overwrite some specific params
        agent_config.memory = "per"
        agent_config.dueling = True 
        agent_config.memory_size = max(50000, agent_config.memory_size)   # For PER we need a bigger buffer  

        # Initialize the parent class
        super().__init__(env, agent_config, is_atari)

        self.name = "Noisy-Dueling-Double-DQN-PER-N-Step"
        self.solved_reward = env_config['solved_reward']
        
        # ---------- PER PARAMS ------------
        self.beta_start = agent_config.beta_start
        self.beta_end = agent_config.beta_final
        self.beta_increase_steps = agent_config.beta_increase_steps
        self.current_beta = self.beta_start
        self.step_count = 0
        
        # ---------- NOISY NET PARAMS ------------
        self.noisy_std = agent_config.noise_std
        
        # Override the policy networks with noisy versions
        self._setup_noisy_networks(agent_config)

    def _setup_noisy_networks(self, agent_config: AgentConfig):
        """Setup noisy networks instead of regular Q-networks"""
        if not self.is_atari:
            if agent_config.dueling:
                self.online_model = NoisyDuelingFC(self.observation_space.shape[0], self.action_space.n, 
                                                   agent_config.hidden_dim, self.noisy_std).to(self.device)
                self.target_model = NoisyDuelingFC(self.observation_space.shape[0], self.action_space.n, 
                                                   agent_config.hidden_dim, self.noisy_std).to(self.device)
            else:
                self.online_model = NoisyFC(self.observation_space.shape[0], self.action_space.n, 
                                            agent_config.hidden_dim, self.noisy_std).to(self.device)
                self.target_model = NoisyFC(self.observation_space.shape[0], self.action_space.n, 
                                            agent_config.hidden_dim, self.noisy_std).to(self.device)
        else:
            if agent_config.dueling:
                self.online_model = NoisyDuelingCNN(self.observation_space.shape, self.action_space.n, 
                                                    agent_config.hidden_dim, self.noisy_std).to(self.device)
                self.target_model = NoisyDuelingCNN(self.observation_space.shape, self.action_space.n, 
                                                    agent_config.hidden_dim, self.noisy_std).to(self.device)
            else:
                self.online_model = NoisyCNN(self.observation_space.shape, self.action_space.n, 
                                             agent_config.hidden_dim, self.noisy_std).to(self.device)
                self.target_model = NoisyCNN(self.observation_space.shape, self.action_space.n, 
                                             agent_config.hidden_dim, self.noisy_std).to(self.device)
        
        # Put the target network in evaluation mode to disable noise
        self.target_model.eval() 
        
        # Copy the initial weights
        self.hard_update_target_network()
        
        # Re-initialize optimizer with new model parameters
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.lr)

    def choose_action(self, state: np.ndarray):
        '''
        Greedy policy with noisy networks.
        No epsilon-greedy needed as exploration is handled by parameter noise.
        '''
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
        # Anneal beta for PER
        progress = min(self.step_count / self.beta_increase_steps, 1.0)
        self.current_beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        
        # 1. Sample a batch of prioritized experiences from PER buffer
        states, actions, rewards, next_states, dones, ns, weights, indices = self.memory.sample(self.batch_size, self.current_beta)
        
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
            # Use online network to select best actions
            next_actions = torch.argmax(self.online_model(next_states), dim=1)
            
            # Use target network to evaluate the Q-values for selected actions
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            
            # Compute n-step return targets
            gamma_n = self.gamma ** ns
            targets = rewards + gamma_n * next_q_values * (1 - dones)
        
        # 5. Calculate TD-errors for PER
        td_errors = torch.abs(q_values - targets)
        
        # 6. Calculate loss with importance sampling weights
        loss = F.mse_loss(q_values, targets, reduction='none')
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
        self.step_count += 1
        
        # 9. Reset noise after all gradient computations are done
        self.reset_noise()
        
        return loss.item()

    def train(self, mean_rewards: List, std_rewards: List, max_steps: int = 100000, mean_n_episodes: int = 50):
        rewards_log = deque(maxlen=mean_n_episodes)
        
        obs, _ = self.env.reset()
        if self.is_atari: obs = self.auto_fire()

        pbar = tqdm(range(max_steps), desc="Training", postfix={"episde": 0, "mean_reward": "N/A", "avg_loss": "N/A"})
        
        episode = 0
        learning_steps = 0

        # Initial noise reset
        self.reset_noise()

        for global_step in pbar:
            action = self.choose_action(obs)
            # No epsilon decay needed for Noisy Networks
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            clipped_reward = self.clip_reward(reward) if self.is_atari else reward
            
            # Add experience to n-step buffer
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
                learning_steps += 1
                if self.should_update_target(learning_steps):
                    self.update_target_network()   

            if done:
                # Episode finished, flush the n-step buffer
                while self.n_step_buffer:
                    n_step_reward, n_step_next_state, n_step_done, n = self._get_n_step_info()
                    start_state, start_action, _, _, _ = self.n_step_buffer.popleft()
                    self.memory.push(start_state, start_action, n_step_reward, n_step_next_state, n_step_done, n)
                
                if "episode" in info: rewards_log.append(info['episode']['r'])
                episode += 1
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
            # Log out the metrics (no epsilon for Noisy Networks)
            postfix = {"episode": episode, "mean_reward": f"{mean_reward:.2f}" if rewards_log else "N/A"}
            pbar.set_postfix(postfix)
        
        pbar.close()