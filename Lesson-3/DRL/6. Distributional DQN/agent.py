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
from core.policies import DistributionalFC, DistributionalCNN, DistributionalDuelingFC, DistributionalDuelingCNN

class AgentDQN(DQNBase):
    '''
    Distributional DQN (C51) algorithm with Prioritized Experience Replay, Double DQN update rule, and N-step returns.
    '''
    def __init__(self, env: gym.Env, agent_config: AgentConfig, env_config: Dict, is_atari: bool):

        # Overwrite some specific params
        agent_config.memory = "per"
        agent_config.dueling = True 
        agent_config.memory_size = max(100000, agent_config.memory_size)   # For PER we need a bigger buffer  

        # Initialize the parent class
        super().__init__(env, agent_config, is_atari)

        self.name = "Distributional-Dueling-Double-DQN-PER-N-Step"
        self.solved_reward = env_config['solved_reward']
        
        # ---------- PER PARAMS ------------
        self.beta_start = agent_config.beta_start
        self.beta_end = agent_config.beta_final
        self.beta_increase_steps = agent_config.beta_increase_steps
        self.current_beta = self.beta_start
        self.step_count = 0
        
        # ---------- C51 PARAMS ------------
        self.n_atoms = agent_config.n_atoms  
        self.v_min = agent_config.v_min  
        self.v_max = agent_config.v_max  
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        
        # Support of the distribution
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)
        
        # Override the policy networks with distributional versions
        self._setup_distributional_networks(agent_config)

    def _setup_distributional_networks(self, agent_config: AgentConfig):
        """Setup distributional networks instead of regular Q-networks"""
        if not self.is_atari:
            if agent_config.dueling:
                self.online_model = DistributionalDuelingFC(self.observation_space.shape[0], self.action_space.n, 
                                                           agent_config.hidden_dim, self.n_atoms).to(self.device)
                self.target_model = DistributionalDuelingFC(self.observation_space.shape[0], self.action_space.n, 
                                                           agent_config.hidden_dim, self.n_atoms).to(self.device)
            else:
                self.online_model = DistributionalFC(self.observation_space.shape[0], self.action_space.n, 
                                                    agent_config.hidden_dim, self.n_atoms).to(self.device)
                self.target_model = DistributionalFC(self.observation_space.shape[0], self.action_space.n, 
                                                    agent_config.hidden_dim, self.n_atoms).to(self.device)
        else:
            if agent_config.dueling:
                self.online_model = DistributionalDuelingCNN(self.observation_space.shape, self.action_space.n, 
                                                            agent_config.hidden_dim, self.n_atoms).to(self.device)
                self.target_model = DistributionalDuelingCNN(self.observation_space.shape, self.action_space.n, 
                                                            agent_config.hidden_dim, self.n_atoms).to(self.device)
            else:
                self.online_model = DistributionalCNN(self.observation_space.shape, self.action_space.n, 
                                                     agent_config.hidden_dim, self.n_atoms).to(self.device)
                self.target_model = DistributionalCNN(self.observation_space.shape, self.action_space.n, 
                                                     agent_config.hidden_dim, self.n_atoms).to(self.device)
        
        # Copy the initial weights
        self.hard_update_target_network()
        
        # Re-initialize optimizer with new model parameters
        self.optimizer = torch.optim.RMSprop(self.online_model.parameters(), lr=self.lr, alpha=0.95, eps=0.01, momentum=0.95, centered=False)

    def choose_action(self, state: np.ndarray):
        '''
        Epsilon-greedy policy for distributional DQN.
        Compute expected Q-values from the distribution.
        '''
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space.n)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            log_probs = self.online_model(state_tensor)  # [1, action_space, n_atoms]
            probs = log_probs.exp()
            
            # Compute expected Q-values: Q(s,a) = sum_i z_i * p_i(s,a)
            q_values = (probs * self.support.view(1, 1, -1)).sum(dim=-1)
            action = torch.argmax(q_values, dim=1).squeeze().item()
            return action

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
        l = b.floor().long()
        u = b.ceil().long()
        
        # Handle the case where l == u (when b is an integer)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (n_atoms - 1)) * (l == u)] += 1
        
        # Initialize projected distribution
        projected_dist = torch.zeros((batch_size, n_atoms), device=self.device)
        
        # Distribute probability mass
        offset = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(batch_size, n_atoms)
        
        # Add probability mass to lower bound
        projected_dist.view(-1).index_add_(
            0, (l + offset * n_atoms).view(-1), 
            (next_probs * (u.float() - b)).view(-1)
        )
        
        # Add probability mass to upper bound
        projected_dist.view(-1).index_add_(
            0, (u + offset * n_atoms).view(-1), 
            (next_probs * (b - l.float())).view(-1)
        )
        
        return projected_dist

    def learn(self):
        """
        Implements learning with Distributional DQN (C51), using L1 loss on expected
        Q-values for PER priority calculation.
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
        
        # 3. Get current state-action distribution
        current_dist_log = self.online_model(states)  # [batch_size, action_space, n_atoms]
        current_dist_log = current_dist_log.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms)).squeeze(1)
        
        # 4. Calculate target distribution using Double DQN
        with torch.no_grad():
            # Use online network to select best actions
            online_next_log_probs = self.online_model(next_states)
            online_next_probs = online_next_log_probs.exp()
            online_next_q = (online_next_probs * self.support.view(1, 1, -1)).sum(dim=-1)
            next_best_actions = torch.argmax(online_next_q, dim=1)
            
            # Use target network to evaluate the distribution for selected actions
            target_next_log_probs = self.target_model(next_states)
            target_next_dist = target_next_log_probs.gather(1, next_best_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms)).squeeze(1)
            
            # Project the target distribution
            projected_dist = self._project_distribution(target_next_dist, rewards, dones, ns)
            
            # Add small epsilon to avoid log(0) in the loss calculation
            projected_dist = projected_dist.clamp(min=1e-8)
        
        # 5. Calculate cross-entropy loss
        loss = -(projected_dist * current_dist_log).sum(dim=-1)
        
        # 6. Calculate TD-errors for PER using L1 loss on expected Q-values
        with torch.no_grad():
            # Expected Q-value of the current state-action pair
            current_q_value = (current_dist_log.exp() * self.support).sum(dim=1)
            
            # Expected Q-value of the target distribution
            target_q_value = (projected_dist * self.support).sum(dim=1)
            
            # TD-error is the absolute difference
            td_errors = torch.abs(current_q_value - target_q_value)
        
        # 7. Apply importance sampling weights
        weighted_loss = weights * loss
        loss = weighted_loss.mean()
        
        # 8. Perform Gradient Descent Step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        # 9. Update priorities in PER
        new_priorities = td_errors.detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, new_priorities)
        
        # Increment step counter for beta annealing
        self.step_count += 1
        
        return loss.item()

    def train(self, mean_rewards: List, std_rewards: List, max_steps: int = 100000, mean_n_episodes: int = 50):
        rewards_log = deque(maxlen=mean_n_episodes)
        
        obs, _ = self.env.reset()
        if self.is_atari: obs = self.auto_fire()

        pbar = tqdm(range(max_steps), desc="Training", postfix={"episde": 0, "mean_reward": "N/A", "avg_loss": "N/A"})
        
        episode = 0
        learning_steps = 0

        for global_step in pbar:
            action = self.choose_action(obs)
            self.decay_epsilon()
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
            # Log out the metrics
            postfix = {"episode": episode, "mean_reward": f"{mean_reward:.2f}" if rewards_log else "N/A", "eps": f"{self.epsilon:.3f}"}
            pbar.set_postfix(postfix)
        
        pbar.close()