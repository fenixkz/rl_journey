import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import gymnasium as gym
from PER import PrioritizedReplayBuffer
from collections import deque

class FCD3QN(nn.Module):
    def __init__(self, obs_space, action_space):
        super(FCD3QN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_space, 128),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean(dim=-1, keepdim=True)
    
class CNND3QN(nn.Module):
    def __init__(self, obs_space, action_space):
        super(CNND3QN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(obs_space[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )   
        feature_size = self.features(torch.zeros(1, *obs_space)).view(1, -1).size(1)
        self.advantage = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )
        
        self.value = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean(dim=-1, keepdim=True)


class D3QN:
    def __init__ (self, 
                  env: gym.Env, 
                  name: str,
                  n_step_return: int, 
                  learning_rate: float = 3e-4, 
                  gamma: float = 0.99, 
                  epsilon: float = 1.0, 
                  epsilon_decay: float = 0.999, 
                  epsilon_min: float = 0.01, 
                  batch_size: int = 32, 
                  buffer_size: int = 100000, 
                  target_update_freq: int = 1000, 
                  validation_period: int = 100,
                  model_type: str = 'FC', 
                  beta_start: float = 0.4, 
                  beta_end: float = 1.0, 
                  beta_annealing_steps: int = 100000, 
                  alpha: float = 0.6) -> None:
        
        self.env = env
        self.name = name
        # Action space dimensions
        self.action_space = env.action_space
        # Learning rate for the optimizer
        self.learning_rate = learning_rate
        # Gamma for discounting future rewards
        self.gamma = gamma
        # Epsilon for exploration
        self.epsilon = epsilon
        # Epsilon decay rate
        self.epsilon_decay = epsilon_decay
        # Minimum epsilon value
        self.epsilon_min = epsilon_min
        # Total number of samples we sample from the replay buffer
        self.batch_size = batch_size
        # How often we update the target network
        self.target_update_freq = target_update_freq
        # Beta for PER
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_annealing_steps = beta_annealing_steps
        # Initialize the replay buffer
        self.buffer = PrioritizedReplayBuffer(buffer_size, alpha)
        # CUDA or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the agents (online and target)
        obs_spec = self.env.observation_space
        action_spec = self.env.action_space
        action_dim_for_model = action_spec.n 
        if model_type == 'FC':
            if isinstance(obs_spec, gym.spaces.Box) and len(obs_spec.shape) == 1:
                obs_input_dim = obs_spec.shape[0]
            elif isinstance(obs_spec, gym.spaces.Discrete): # e.g. if obs is a single number
                obs_input_dim = 1 
            else: # E.g. Box with image dimensions, needs flattening
                obs_input_dim = np.prod(obs_spec.shape)
            self.online_model = FCD3QN(obs_input_dim, action_dim_for_model).to(self.device)
            self.target_model = FCD3QN(obs_input_dim, action_dim_for_model).to(self.device)
        elif model_type == 'CNN':
            # CNND3QN expects obs_space to be the shape tuple (e.g., (C, H, W))
            # It internally uses obs_space[0] for channels.
            self.online_model = CNND3QN(self.env.observation_space.shape, action_dim_for_model).to(self.device)
            self.target_model = CNND3QN(self.env.observation_space.shape, action_dim_for_model).to(self.device)
        else:
            raise ValueError("Invalid model type. Choose 'FC' or 'CNN'.")
        # Copy the weights from the online model to the target model
        self.update_target(total_steps=0)
        # Optimizer for the online model
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.learning_rate)
        # How often we validate the model
        self.validation_period = validation_period
        # Number of steps for n-step return
        self.n_step_return = n_step_return
        print("D3QN agent initialized")


    def update_target(self, total_steps: int) -> None:
        """Update the target model with the online model's weights."""
        if total_steps % self.target_update_freq == 0:
            # Update the target model with the online model's weights
            self.target_model.load_state_dict(self.online_model.state_dict())

    def get_beta(self, total_steps: int) -> float:
        """Calculates the current beta for PER annealing."""
        fraction = min(total_steps / self.beta_annealing_steps, 1.0)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)
    
    def choose_action(self, state: np.ndarray, epsilon: float) -> int:
        '''
        Choose an action with epsilon-greedy policy.
        '''
        if np.random.random() < epsilon:
            return np.random.choice(self.action_space.n)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(self.device)
                q_values = self.online_model(state_tensor)
                action = torch.argmax(q_values, dim=1).squeeze().item() 
                return action
            
    def learn(self, n_steps: int) -> float:
        # 1. Sample batch from PER buffer
        # Returns: state, action, reward, next_state, done, weights, indices
        state, action, reward, next_state, done, ns, weights, indices = self.buffer.sample(self.batch_size, self.get_beta(n_steps))
        # 2. Convert to tensors and move to device
        state = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device) 
        action = torch.LongTensor(action).to(self.device) # Actions are indices, should be Long
        reward = torch.FloatTensor(reward).to(self.device)
        # done flags are often treated as floats (0.0 or 1.0) for masking
        done = torch.FloatTensor(done.astype(np.float32)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device) # Importance sampling weights
        ns = torch.LongTensor(ns).to(self.device) # Number of steps for n-step return

        # 3. Calculate Q(s, a) for the actions taken
        # Get Q-values for all actions from online model
        q_values_all_actions: torch.Tensor = self.online_model(state) # Shape: [batch_size, num_actions]
        # Use gather to select the Q-value for the specific action taken
        q_s_a = q_values_all_actions.gather(1, action.unsqueeze(-1)).squeeze(-1) # Shape: [batch_size, 1]

        # 4. Calculate TD-target
        with torch.no_grad(): # Target calculations should not affect gradients
            # --- DDQN Action Selection ---
            # Use online model to find the best action indices in the next state
            online_next_q_values = self.online_model(next_state)
            next_action_indices = torch.argmax(online_next_q_values, dim=1) # Shape: [batch_size, 1]

            # --- DDQN Action Evaluation ---
            # Use target model to get Q-values for all actions in the next state
            target_next_q_values_all = self.target_model(next_state)
            # Select the Q-value from the target network corresponding to the action chosen by the online network
            q_s_prime_a_prime = target_next_q_values_all.gather(1, next_action_indices.unsqueeze(-1)).squeeze(-1) # Shape: [batch_size, 1]

            # --- Calculate TD Target ---
            # Target = r + gamma * Q_target(s', argmax_a' Q_online(s', a')) * (1 - done)
            # Multiply by (1 - done) so target is just 'r' if next_state is terminal
            td_target = reward + self.gamma**ns * q_s_prime_a_prime * (1 - done)

        # 5. Calculate Loss 
        # Calculate TD Error: td_target - q_s_a
        td_error = td_target - q_s_a

        # Calculate loss: (TD Error)^2 weighted by IS weights
        # Using detach() on td_target ensures gradients only flow through q_s_a
        loss: torch.Tensor = (weights * (td_error.pow(2))).mean()

        # 6. Update Priorities in PER Buffer
        priorities = torch.abs(td_error.detach()) + 1e-6 # Small epsilon to avoid zero priorities
        self.buffer.update_priorities(indices, priorities.cpu().numpy())

        # 7. Perform Gradient Descent Step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self) -> float:
        rewards = []
        for i in range(10):
            state = self.env.reset()[0]
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state, epsilon=0.0)  # Epsilon = 0 for validation
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward
            rewards.append(total_reward)
        return np.mean(rewards)

    def train(self, n_episodes: int):
        # Store the history of total epsiode rewards
        test_rewards = []
        train_rewards = []
        total_steps = 0
        best_test_reward = -float('inf')
        for e in range(n_episodes):
            state = self.env.reset()[0]
            episode_reward = 0
            episode_loss = 0
            learning_steps = 0
            terminated, truncated = False, False
            while not terminated and not truncated:
                # For N-step return, initialize the buffer
                n_return_buffer = []
                for i in range(self.n_step_return):
                    total_steps += 1
                    # Choose action given our epsilon greedy policy
                    action = self.choose_action(state, self.epsilon)
                    # Step
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    exp = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminated': terminated}
                    n_return_buffer.append(exp)
                    state = next_state
                    episode_reward += reward
                    if terminated or truncated:
                        break
                # Calculate n-step return
                G = 0
                for exp in reversed(n_return_buffer):
                    G = exp['reward'] + self.gamma * G
        
                # Record data into buffer
                self.buffer.push(
                                n_return_buffer[0]['state'],          # S_t
                                n_return_buffer[0]['action'],         # A_t
                                G,                                    # N-step discounted return
                                n_return_buffer[-1]['next_state'],    # S_{t+actual_n}
                                n_return_buffer[-1]['terminated'],    # done flag for S_{t+actual_n}
                                len(n_return_buffer)                  # actual_n
                                )
                # Update target model if time has come
                self.update_target(total_steps)
                # If enough samples, then backpropogate and learn
                if len(self.buffer) > self.batch_size:
                    loss = self.learn(total_steps)
                    learning_steps += 1
                    episode_loss += loss
            # Decay epsilon
            self.epsilon  = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            avg_loss = episode_loss / learning_steps if learning_steps > 0 else 0
            print(f"Episode {e}/{n_episodes}, Episode Reward: {episode_reward}, Episode Mean loss: {avg_loss}, Current Epsilon: {self.epsilon}")
            if e % self.validation_period == 0:
                with torch.no_grad():
                    test_reward = self.validate()
                    print(f"Mean validation reward: {test_reward}")
                    test_rewards.append(test_reward)
                if test_reward > best_test_reward:
                    # Save the online model after training
                    model_save_path = f"{self.name}_best_checkpoint.pth"
                    checkpoint = {
                        'model_state_dict': self.online_model.state_dict(),
                        'epsilon': self.epsilon,
                        'episode': e,
                        'best_test_reward': best_test_reward
                    }
                    torch.save(checkpoint, model_save_path)
                    
                    print(f"Best model saved to {model_save_path}")
                    best_test_reward = test_reward
            train_rewards.append(episode_reward)
        return train_rewards, test_rewards