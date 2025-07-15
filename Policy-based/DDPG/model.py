import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random
from torch.optim import AdamW
import os

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class Actor(nn.Module):
    '''
    Actor accepts state as input and returns a single action per action_dimension.
    '''
    def __init__(self, state_dim, action_dim: int, upper: float, lower: float, device: str):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh() # Squash the output to the [-1, 1]
        )
        self.upper_threshold = upper
        self.lower_threshold = lower
        self.range = torch.from_numpy(upper - lower).view(1, action_dim).to(device) / 2.0
        self.midpoint = torch.from_numpy((upper+lower) / 2.0).view(1, action_dim).to(device)
        
    def forward(self, state: torch.Tensor):
        action = self.model(state) * self.range # Given the output tensor of shape (B, N-actions) and range shape: (1, N), via broadcasting result is (B, N-actions)
        action = action + self.midpoint
        return action 

class Critic(nn.Module):
    '''
    Critic accepts state and action as input and returns a Q-values of each state-action pair, Works only for vector observations. 
    If the observation is images, then this architecture won't work. 
    '''
    def __init__(self, state_dim, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim+action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        '''
        Assuming state and action are Tensors with batch dimension
        '''
        features = torch.concat((state, action), dim = 1)
        q_value = self.model(features)
        return q_value
    


class DDPGAgent:

    def __init__(self, obs_space, action_space, noise_magnitude: float = 0.1, noise_decay: float = 0.9995, min_noise: float = 0.01, gamma: float = 0.99, actor_lr: float = 1e-4, critic_lr: float = 3e-4, 
                 batch_size: int = 256, tau: float = 0.005):
        
        # Correctly parse space dimensions
        self.state_dim = obs_space.shape[0]
        self.action_dim = action_space.shape[0]
        
        # Get a device for training 
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Initialize online and target networks
        self.online_actor = Actor(self.state_dim, self.action_dim, action_space.high, action_space.low, self.device).to(self.device)
        self.online_critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_actor = Actor(self.state_dim, self.action_dim, action_space.high, action_space.low, self.device).to(self.device)
        self.target_critic = Critic(self.state_dim, self.action_dim).to(self.device)
        
        # Copy online weights to target networks
        self.target_actor.load_state_dict(self.online_actor.state_dict())
        self.target_critic.load_state_dict(self.online_critic.state_dict())
        
        # Initialize noise parameters for exploration
        self.noise_decay = noise_decay
        self.min_noise_magnitude = min_noise
        self.noise_magnitude = noise_magnitude
        
        # Get the range of actions for clamping
        self.action_low = torch.FloatTensor(action_space.low).to(self.device)
        self.action_high = torch.FloatTensor(action_space.high).to(self.device)
        
        # Set learning parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        
        # Losses and Optimizers
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = AdamW(self.online_actor.parameters(), lr=actor_lr)
        self.critic_optimizer = AdamW(self.online_critic.parameters(), lr=critic_lr)

    def get_action(self, state, apply_noise: bool = True) -> np.ndarray:
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.online_actor(state)
        
            if apply_noise:
                noise = torch.randn_like(action) * self.noise_magnitude
                action += noise
        
        # Detach from graph, move to CPU, convert to NumPy, and remove batch dimension
        return torch.clamp(action, self.action_low, self.action_high).detach().cpu().numpy().squeeze(0)
    
    def decay_noise(self):
        if self.noise_magnitude > self.min_noise_magnitude:
            self.noise_magnitude *= self.noise_decay

    def soft_update(self):
        """Soft update the target networks with the online networks' weights using tau."""
        for target_param, online_param in zip(self.target_critic.parameters(), self.online_critic.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)
            
        for target_param, online_param in zip(self.target_actor.parameters(), self.online_actor.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    def learn(self, batch):
        state, action, reward, next_state, done = batch
        # Convert all entities into tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).view(self.batch_size, 1) # Good to be explicit with shapes
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.LongTensor(done).to(self.device).view(self.batch_size, 1) # Good to be explicit with shapes

        # DPG update rule:
        # First get the actions for next states using target actor, note we are not using exploration noise
        with torch.no_grad(): # All operations done by target network should not be tracked!
            next_actions = self.target_actor(next_state)
            # Second get Q-values of next-state next-actions 
            next_qs = self.target_critic(next_state, next_actions)
            target = reward + self.gamma * next_qs * (1 - done)
        # Get Q-values of current states action pairs
        q_values = self.online_critic(state, action)
        # Finally apply MSE loss to get critic loss
        critic_loss = self.critic_criterion(q_values, target)
        # Order matter here, first the critic has to be updated, because actor gradients depend on the critic gradients wrt to action. So, we should update critic's weight before updating actor's weights
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update Rule
        """
        This part has to be understood correctly: the actor just chooses an action based on the state
        So, at some point of data collection we had an old actor (hopefully less trained and worse), that actor was picking an action for that state. 
        Now we have a better actor, because hopefully, our training made it better. Why would we reinforce the actor based on the decision it made when it was dumber? 
        The more profitable approach would be to ask the new actor what would it do given that state. Then reinforce it based on this action and not the old one
        """
         
        new_actions = self.online_actor(state)
        new_q_values = self.online_critic(state, new_actions)
        actor_loss = -1 * new_q_values.mean()

        
        # Then the actor
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

        # Soft update the target networks
        self.soft_update()
        
    def save_models(self, env_name="ddpg_agent"):
        """Saves the state dictionaries of the online actor and critic."""
        os.makedirs("models", exist_ok=True)

        print("--- Saving models ---")
        torch.save(
            self.online_actor.state_dict(),
            f"models/{env_name}_actor.pth"
        )
        torch.save(
            self.online_critic.state_dict(),
            f"models/{env_name}_critic.pth"
        )

    def load_models(self, env_name="ddpg_agent"):
        """Loads the state dictionaries for the online actor and critic."""
        print("--- Loading models ---")
        self.online_actor.load_state_dict(
            torch.load(f"models/{env_name}_actor.pth")
        )
        self.online_critic.load_state_dict(
            torch.load(f"models/{env_name}_critic.pth")
        )
        # It's good practice to also load these into the target networks
        # to ensure a consistent starting point for further training or evaluation
        self.target_actor.load_state_dict(self.online_actor.state_dict())
        self.target_critic.load_state_dict(self.online_critic.state_dict())