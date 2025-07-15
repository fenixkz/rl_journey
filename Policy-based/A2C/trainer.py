import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import os
from agent import A2CAgent
import gymnasium as gym
from collections import deque

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class A2CTrainer:
    """
    A2C (Advantage Actor-Critic) trainer with vector environments
    """
    
    def __init__(self, 
                env_name='CartPole-v1',
                num_envs=8,
                num_steps=5, 
                actor_lr=3e-4,
                critic_lr=1e-3, 
                gamma=0.99,
                gae_lambda=0.95,
                entropy_coef=0.01,
                hidden_size=64,
                max_steps=200000, 
                solved_threshold=495.0, 
                device='cpu',
                seed=42,
                normalize_advantage=False):
        
        self.env_name = env_name
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.max_steps = int(max_steps)
        self.solved_threshold = solved_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.normalize_advantage = normalize_advantage

        set_seed(seed)
        
        env_fns = [lambda: gym.make(env_name) for _ in range(num_envs)]
        self.envs = gym.vector.SyncVectorEnv(env_fns)
        
        obs_dim = self.envs.single_observation_space.shape[0]
        action_dim = self.envs.single_action_space.n
        
        self.agent = A2CAgent(obs_dim, action_dim, hidden_size, self.device, 
                              actor_lr, critic_lr, self.entropy_coef).to(self.device)
        
        # Pre-allocated buffers for data that does NOT need a computation graph
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.current_states = None # To keep track of current states in transition between rollouts

        self.current_episode_rewards = np.zeros(self.num_envs, dtype=np.float64)
        self.env_rewards = {i: [] for i in range(self.num_envs)}
        self.episode_rewards = []
        self.total_steps = 0

    def train(self):
        states, _ = self.envs.reset(seed=self.seed)
        self.current_states = states
        while self.total_steps < self.max_steps:
            # Collect rollouts and get back the tensors with computation graphs
            log_probs, values, returns, advantages, entropies = self.collect_rollouts()
            
            # Update the agent
            self.agent.learn(values, returns, advantages, log_probs, entropies)
            
            # Update total steps
            self.total_steps += self.num_envs * self.num_steps
            if np.mean(self.episode_rewards[-100:]) > self.solved_threshold:
                print("Env has been solved!")
                break
        return self.episode_rewards

    def compute_returns_and_advantages(self, values, next_values):
        """
        Compute returns and advantages using GAE from collected rollout data.
        Args:
            values: Values from the rollout with computation graph.
            next_values: Values for the states after the last rollout step.
        Returns:
            returns: N-step returns (targets for the critic).
            advantages: GAE advantages (weights for the actor).
        """
        detached_values = values.detach()
        advantages = torch.zeros_like(self.rewards).to(self.device)
        
        last_gae_lambda = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_vals = next_values
            else:
                next_vals = detached_values[t + 1]
            
            next_non_terminal = 1.0 - self.dones[t] 
            delta = self.rewards[t] + self.gamma * next_vals * next_non_terminal - detached_values[t]
            advantages[t] = last_gae_lambda = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
        
        returns = advantages + detached_values
        return returns, advantages
    
    def collect_rollouts(self):
        """
        Collects `num_steps` of experience from each environment.
        """
        # Lists to temporarily hold tensors with computation graphs
        log_probs_list = []
        values_list = []
        entropies_list = []
        current_states = self.current_states
        for i in range(self.num_steps):
            # Get action and value from agent
            probs, values = self.agent.act_and_evaluate(current_states)
            distribution = torch.distributions.Categorical(probs)
            actions = distribution.sample()
            
            # Step in the environments
            next_states, rewards, terminated, truncated, _ = self.envs.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated)
            
            # Store data
            log_probs_list.append(distribution.log_prob(actions))
            values_list.append(values.squeeze())
            entropies_list.append(distribution.entropy())
            self.rewards[i] = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            self.dones[i] = torch.tensor(dones, dtype=torch.float32).to(self.device)
            
            current_states = next_states
            self.current_episode_rewards += rewards
            for env_idx, done in enumerate(dones):
                if done:
                    # An episode finished in this environment.
                    # 1. Log the completed episode's total reward.
                    final_reward = self.current_episode_rewards[env_idx]
                    self.env_rewards[env_idx].append(final_reward)
                    if env_idx == 0:
                        print(f"[ENV {env_idx}] Episode {len(self.env_rewards[env_idx])}. Total Reward: {final_reward:.2f}")
                    self.episode_rewards.append(final_reward)
                    # 2. Reset the trackers for this specific environment.
                    self.current_episode_rewards[env_idx] = 0
        # Save global object of the current states, such that next rollout knows where to start
        self.current_states = current_states
        # After the loop, stack the collected tensors
        log_probs = torch.stack(log_probs_list)
        values = torch.stack(values_list)
        entropies = torch.stack(entropies_list)

        # Bootstrap value for GAE
        with torch.no_grad():
            next_values = self.agent.evaluate(next_states).squeeze()
        
        # Calculate returns and advantages
        returns, advantages = self.compute_returns_and_advantages(values, next_values)
        
        # Flatten all tensors for the learning step
        log_probs_flat = log_probs.view(-1)
        values_flat = values.view(-1)
        returns_flat = returns.view(-1)
        advantages_flat = advantages.view(-1)
        entropies_flat = entropies.view(-1)
        
        # Normalize advantages (important for stability)
        if self.normalize_advantage:
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        return log_probs_flat, values_flat, returns_flat, advantages_flat, entropies_flat

