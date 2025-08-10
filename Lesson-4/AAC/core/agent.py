import torch
import torch.nn as nn
import torch.nn.functional as F
from core.actor_critic import Actor, Critic
import gymnasium as gym
import numpy as np
import random 
from tqdm import tqdm
from collections import deque
from typing import List, Tuple
import os 

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class AACAgent:

    def __init__(self, 
                 env: gym.Env,
                 solved_threshold: int,
                 gamma: float = 0.99,
                 hidden_size: int = 64,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 seed: int = 24,
                 use_normalization: bool = False,
                 use_entropy: bool = False,
                 entropy_coef: float = 0.01):
        assert len(env.observation_space.shape) == 1, "REINFORCE supports for now only Box2D envs"
        self.env = env
        self.solved_threshold = solved_threshold
        self.gamma = gamma
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.use_normalization = use_normalization
        self.use_entropy = use_entropy
        self.entropy_coef = entropy_coef
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        set_seed(seed)
        self.actor = Actor(self.observation_space.shape[0], self.action_space.n, hidden_size).to(self.device)
        self.critic = Critic(self.observation_space.shape[0], hidden_size).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        env.reset(seed = seed)
        env.action_space.seed(seed)

    def act(self, state: np.ndarray) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        # State is internally converted to a tensor by the actor
        # Logits are the raw scores for each action
        logits = self.actor(state) # shape: (batch_size, action_space)
        
        # Apply softmax to get probabilities
        # Convert logits to probabilities via softmax operator
        probs = F.softmax(logits, dim = -1) # shape: (batch_size, action_space)
        
        # Create a categorical distribution from the probabilities
        # This distribution will be used to sample actions and compute log probabilities
        action_dist = torch.distributions.Categorical(probs) # Categorical distribution for sampling actions, shape: (batch_size, action_space)

        # Sample an action from the distribution
        # action_dist.sample() returns a tensor of shape (batch_size,) with the sampled actions
        # We use item() to get the scalar value from the tensor
        action = action_dist.sample() # Sample an action from the distribution, shape: (batch_size,)
        return action_dist, action 
    
    def evaluate(self, state: np.ndarray) -> torch.Tensor:
        return self.critic(state) # State is internally converted to a tensor by the critic

    def play_one_episode(self):
        state, _ = self.env.reset()
        done = False
        
        # Data structures to store all data
        states, next_states, rewards, dones, log_probs, entropies = [], [], [], [], [], []


        while not done:
            action_dist, action = self.act(state)
            # Apply the action to the environment
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())  # Use .item() to extract the integer from the tensor
            done = terminated or truncated
            
            # Store log probabilities and rewards
            states.append(state)  # Store the current state
            next_states.append(next_state)  # Store the next state
            rewards.append(reward)
            dones.append(done)
            log_probs.append(action_dist.log_prob(action))
            if self.use_entropy: entropies.append(action_dist.entropy())

            # Transition to the next state
            state = next_state 

        return states, next_states, rewards, dones, log_probs, entropies 

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards): # Use backward pass to compute returns for each step
            G = r + self.gamma * G
            returns.insert(0, G)
        
        # Convert to tensors
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def train(self, all_rewards: List, max_episodes: int = 2000):
        rewards_log = deque(maxlen=100)
        pbar = tqdm(range(max_episodes), desc=f"Training")
        for e in pbar:
            states, next_states, rewards, dones, log_probs, entropies  = self.play_one_episode()

            # Compute metrics and check if environment is solved before casting to tensors
            avg_reward = -float('inf')
            if len(rewards_log) > 10: avg_reward = np.mean(rewards_log) # to avoid computing mean of empty deque
            
            if avg_reward >= self.solved_threshold:
                pbar.write(f"🎉 Environment solved at episode {e}!")
                break  # Stop training if solved
            
            # Print progress
            pbar.set_postfix({
                                'Mean': f"{avg_reward:.1f}",
                            })
            

            rewards_log.append(sum(rewards))
            all_rewards.append(sum(rewards))

            # Convert lists to tensors
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)  # Convert rewards to tensor, shape: [n_steps, 1]
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)   # Convert dones to tensor, shape: [n_steps, 1]
            log_probs = torch.stack(log_probs).to(self.device)  # Stack log probabilities into a tensor, shape: [n_steps, 1]
            entropies = torch.stack(entropies).to(self.device)  # Stack entropies into a tensor, shape: [n_steps, 1]
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)  # Convert states to tensor, shape: [n_steps, state_dim]
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)  # Convert next states to tensor, shape: [n_steps, state_dim]
            
            # Estimate state values
            state_values = self.evaluate(states)

            # Compute TD-target using the 1 step return
            with torch.no_grad():
                v_next = self.evaluate(next_states)
                # For terminal states, next value should be 0
                td_target = rewards + self.gamma * v_next * (1 - dones)

            advantage = td_target - state_values  # advantage shape: [N, 1] (tensor)
            # Normalize advantage if specified
            if self.use_normalization: advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                
            # Compute losses
            # Actor loss: -log_prob * A(s, a).
            # Detach advantage for actor loss, the gradients must flow only through log_probs
            actor_loss = -(log_probs * advantage.detach()).mean()
            if self.use_entropy: actor_loss -=  self.entropy_coef*entropies.mean()
            # Critic loss: L2 between target and state values
            critic_loss = F.mse_loss(state_values, td_target.detach())  

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1) # For stability
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1) # For stability
            self.critic_optimizer.step()

        self.env.close()
        pbar.close()
    
    def save_policy(self, save_path: str):
        '''
        Save policy for later evaluation
        '''
        os.makedirs(save_path, exist_ok=True)
        actor_save_path = os.path.join(save_path, "actor.pth")
        critic_save_path = os.path.join(save_path, "critic.pth")
        torch.save(self.actor.state_dict(), actor_save_path)
        torch.save(self.actor.state_dict(), critic_save_path)
        
    def load_policy(self, load_path: str):
        '''
        Load pre-trained policy
        '''
        os.makedirs(load_path, exist_ok=True)
        actor_save_path = os.path.join(load_path, "actor.pth")
        critic_save_path = os.path.join(load_path, "critic.pth")
        self.actor.load_state_dict(torch.load(actor_save_path))
        self.critic.load_state_dict(torch.load(critic_save_path))