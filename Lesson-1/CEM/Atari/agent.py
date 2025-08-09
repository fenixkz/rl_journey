import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import os
import gymnasium as gym
import random
from tqdm import tqdm
from collections import deque
from typing import List
import multiprocessing as mp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.atari_utils import get_atari_env, clip_reward

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

class FCNetwork(nn.Module):
    '''
    A fully-connected neural network. This is our policy, $\pi(\theta)$. 
    Given the state, returns the probability distribution over actions.
    '''
    
    def __init__(self, obs_space, action_space, hidden_space: int = 128):
        super(FCNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space),
        )
        self._init_weights()
        
    def forward(self, x: torch.Tensor):
        return self.network(x)
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

class CNNNetwork(nn.Module):
    '''
    A convolutional neural network for environments with states as images. 
    This is our policy, $\pi(\theta)$. 
    Given the state, returns the probability distribution over actions.
    '''
    def __init__(self, obs_space, action_space, hidden_space: int = 128):
        super(CNNNetwork, self).__init__()
        C,H,W = obs_space
        self.backbone = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Pass a zero tensor to get the final flattened shape of the resulting tensor
        with torch.no_grad():
            feature_size = self.backbone(torch.zeros(1, *obs_space)).view(1, -1).size(1)
        self.regressor = nn.Sequential(
            nn.Linear(feature_size, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space)
        )
        self._init_weights()

    def forward(self, x: torch.Tensor):
        # Extract features from the image
        x = self.backbone(x)
        # Flatten
        x = x.view(x.size(0), -1)
        return self.regressor(x)
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization for conv layers and Xavier for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

def evaluate_episode(args):
    """
    Asynchronous function for multiprocessing. Each process runs one episode.
    Returns the history (state-action pairs) and total reward for that episode.
    """
    # Unpack the arguments
    policy_state_dict, env_id, seed, hidden_dim, is_atari, random, add_action_noise = args
    
    # Each process needs its own environment and agent instance
    env = gym.make(env_id) if not is_atari else get_atari_env(env_id)
    # Use a different seed for each worker's env to ensure diverse episodes
    env.reset(seed=seed)
    env.action_space.seed(seed)
    
    # Recreate the policy network structure
    agent = EnhancedCEMAgent(env, solved_threshold=9999, is_atari=is_atari, hidden_dim=hidden_dim)
    
    # Load the policy weights
    agent.load_state_dict(policy_state_dict)
    
    # Play the episode and collect data
    history, total_reward = agent.play_one_episode(random=random, add_action_noise=add_action_noise)
    
    # Extract observations and actions from history
    episode_observations = [step[0] for step in history]
    episode_actions = [step[1] for step in history]
    
    env.close()
    return (episode_observations, episode_actions, total_reward)

class EnhancedCEMAgent(nn.Module):
    '''
    This is our Agent with a policy represented by a neural network. 
    The network accepts the state as the input and outputs a probability distribution over actions. 
    '''
    def __init__(self, 
                 env: gym.Env,
                 solved_threshold: float,
                 is_atari: bool = False,
                 hidden_dim: int = 64,
                 lr: float = 1e-3,
                 seed: int = 224,
                 cpu_usage: float = 0.5,
                 # New Atari-specific parameters
                 use_warm_start: bool = True,
                 warm_start_episodes: int = 30,
                 warm_start_epsilon: float = 0.3,
                 use_adaptive_percentile: bool = True,
                 min_percentile: int = 30,
                 use_noise_injection: bool = True,
                 noise_std: float = 0.1):
        """
        Initialize EnhancedCEMAgent with exploration enhancements.
        
        Args:
            env: Atari gymnasium environment
            solved_threshold: Reward threshold for considering the environment solved
            hidden_dim: Hidden layer dimension for CNN
            lr: Learning rate
            seed: Random seed
            cpu_usage: Fraction of CPU cores to use for multiprocessing
            use_warm_start: Whether to use warm-start exploration phase
            warm_start_episodes: Number of episodes for warm-start
            warm_start_epsilon: Epsilon for epsilon-greedy during warm-start
            use_adaptive_percentile: Whether to dynamically adjust percentile
            min_percentile: Minimum percentile when using adaptive adjustment
            use_noise_injection: Whether to add noise when all rewards are identical
            noise_std: Standard deviation for noise injection
        """
        assert(isinstance(env.action_space, gym.spaces.Discrete)), "Detected non-discrete action space, this class works only with discrete action space problems!"
        set_seed(seed)
        super(EnhancedCEMAgent, self).__init__()

        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.solved_threshold = solved_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.is_atari = is_atari
        self.hidden_dim = hidden_dim
        self.use_mp = is_atari  # Use multiprocessing for Atari environments only
        self.cpu_usage = cpu_usage


        if not is_atari:
            # print(f"Detected a classic (vector) environment, observation space shape: {self.observation_space.shape}, using Fully Connected (FC) Network")
            self.policy = FCNetwork(self.observation_space.shape[0], self.action_space.n, hidden_dim).to(self.device)
        else:
            # print(f"Detected an Atari environment, observation space shape: {self.observation_space.shape}, using Convolutional Neural Network (CNN)")
            self.policy = CNNNetwork(self.observation_space.shape, self.action_space.n, hidden_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        
        # First reset for reproducing same results
        env.reset(seed = seed)
        env.action_space.seed(seed)

        # Extra parameters for enhancement in exploration
        self.use_warm_start = use_warm_start
        self.warm_start_episodes = warm_start_episodes
        self.warm_start_epsilon = warm_start_epsilon
        self.use_adaptive_percentile = use_adaptive_percentile
        self.min_percentile = min_percentile
        self.base_percentile = None  # Will be set during training
        self.use_noise_injection = use_noise_injection
        self.noise_std = noise_std
        self.performance_history = deque(maxlen=10)  # Track recent performance

    def choose_action(self, obs: np.ndarray, random: bool = False, add_action_noise: bool = False) -> np.ndarray:
        '''
        A function that decides what action to choose given the state. Can be deterministic (choosing action with the highest probability) or stochastic (sampling action from a probability distribution). 
        '''
        if random:
            if np.random.random() < self.warm_start_epsilon:
                return self.env.action_space.sample()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device) # Add a batch dimension
        # Forward pass, get logits
        logits = self.policy.forward(obs)

        if add_action_noise and self.use_noise_injection:
            noise = torch.randn_like(logits) * self.noise_std
            logits += noise
        # Sample
        probs = F.softmax(logits, dim=-1) # Convert logits to probabilities
        dist = Categorical(probs)         # Create a distribution object
        action = dist.sample()            # Sample an action
        # Return as numpy scalar or array (remove batch dim)
        return action.item()

    def select_elite_episodes(self, episodes_data, percentile):
        """
        Enhanced elite selection with tie-breaking for sparse rewards.
        
        Args:
            episodes_data: List of (states, actions, total_reward) tuples
            percentile: Percentile threshold for elite selection
            
        Returns:
            Tuple of (elite_observations, elite_actions) as numpy arrays
        """
        # Sort episodes by total reward
        episodes_data.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate elite count with minimum guarantee
        n_elite = int(len(episodes_data) * (100 - percentile) / 100)
        
        # Get elite episodes
        elite_episodes = episodes_data[:n_elite]
        
        # Extract observations and actions
        elite_obs = []
        elite_actions = []
        
        for states, actions, _ in elite_episodes:
            elite_obs.extend(states)
            elite_actions.extend(actions)
        
        return np.array(elite_obs), np.array(elite_actions)

    def calculate_adaptive_percentile(self, episodic_rewards):
        """
        Dynamically adjust percentile based on performance variance.
        
        Args:
            episodic_rewards: List of episode rewards
            
        Returns:
            Adjusted percentile value
        """
        if not self.use_adaptive_percentile:
            return self.base_percentile
        
        reward_std = np.std(episodic_rewards)
        reward_range = max(episodic_rewards) - min(episodic_rewards)
        
        # Track performance improvement
        mean_reward = np.mean(episodic_rewards)
        self.performance_history.append(mean_reward)
        
        # Check if we're stuck (no improvement in recent history)
        if len(self.performance_history) >= 5:
            recent_improvement = max(self.performance_history) - min(self.performance_history)
            is_stuck = recent_improvement < 0.5
        else:
            is_stuck = False
        
        # Adjust percentile based on variance and progress
        if reward_std < 0.5 or reward_range < 1.0 or is_stuck:
            # Low variance or stuck - be more inclusive
            adaptive_percentile = max(self.min_percentile,
                                     self.base_percentile - 30)
            print(f"Low variance/stuck, using adaptive percentile: {adaptive_percentile}% "
                  f"(std={reward_std:.2f}, range={reward_range:.2f})")
        elif reward_std > 2.0 and reward_range > 5.0:
            # High variance - can be more selective
            adaptive_percentile = min(90, self.base_percentile + 10)
            print(f"High variance, using adaptive percentile: {adaptive_percentile}%")
        else:
            # Normal variance - use base percentile
            adaptive_percentile = self.base_percentile
        
        return adaptive_percentile

    def warm_start_exploration(self):
        """
        Collect diverse initial episodes using epsilon exploration.
        
        Returns:
            List of (observations, actions, reward) tuples from warm-start episodes
        """
        print(f"\n{'='*60}")
        print(f"Running warm-start exploration for {self.warm_start_episodes} episodes...")
        print(f"Using epsilon={self.warm_start_epsilon} for exploration")
        print(f"{'='*60}")
        
        warm_data = []
        
        for _ in range(self.warm_start_episodes):

            history, total_reward = self.play_one_episode(random=True)
            # Extract observations and actions from history
            episode_observations = [step[0] for step in history]
            episode_actions = [step[1] for step in history]
    
            warm_data.append((episode_observations, episode_actions, total_reward))
        
        return warm_data
    
    def warm_start_with_multiprocessing(self):
        """
        Parallel version of warm-start exploration using multiprocessing.
        """
        print(f"\n{'='*60}")
        print(f"Running parallel warm-start exploration for {self.warm_start_episodes} episodes...")
        print(f"Using epsilon={self.warm_start_epsilon} for exploration")
        print(f"{'='*60}")
        
        # Get current policy state for sharing with workers
        policy_state_dict = self.state_dict()
        
        # Create tasks for multiprocessing
        tasks = []
        for i in range(self.warm_start_episodes):
            task_args = (
                policy_state_dict,
                self.env.spec.id,
                i * 1000,  # Different seed for each episode
                self.hidden_dim,
                self.is_atari,
                True,  # random=True for exploration
                False, # add_action_noise
            )
            tasks.append(task_args)
        
        # Run episodes in parallel
        with mp.Pool(processes=int(os.cpu_count() * self.cpu_usage)) as pool:
            warm_data = pool.map(evaluate_episode, tasks)
        
        # Extract and print statistics
        warm_rewards = [episode[2] for episode in warm_data]
        print(f"\nWarm-start complete!")
        print(f"  Reward range: [{min(warm_rewards):.1f}, {max(warm_rewards):.1f}]")
        print(f"  Mean reward: {np.mean(warm_rewards):.2f}")
        print(f"  Std reward: {np.std(warm_rewards):.2f}")
        
        return warm_data

    def learn(self, observations, actions):
        '''
        A function that trains the network to better predict actions from given states. 
        Uses CrossEntropyLoss
        '''
        # Convert to tensors 
        obs_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        # Get predictions
        pred_actions = self.policy.forward(obs_tensor)
        # Backward pass
        self.optimizer.zero_grad()
        loss = self.loss(pred_actions, actions_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_policy(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        print("--- Saving Policy ---")
        torch.save(self.state_dict(), f"{save_path}/policy.pth")
        print(f"Policy saved in {save_path}")

    def load_policy(self, env_name: str):
        load_path = f"results/{env_name}"
        try:
            self.load_state_dict(
                torch.load(f"{load_path}/policy.pth")
            )
        except FileNotFoundError:
            print(f"Policy file not found in {load_path}. Using untrained policy.")

    def play_one_episode(self, random: bool = False, add_action_noise: bool = False):
        obs, _ = self.env.reset()
        history = []
        done = False
        total_reward = 0
        while not done:
            action = self.choose_action(obs=obs, random=random, add_action_noise = add_action_noise)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            if self.is_atari: reward = clip_reward(reward=reward)
            history.append((obs, action))
            obs = next_obs
            total_reward += reward
        return history, total_reward
        
    def train(self, all_rewards: List, num_epochs: int, num_episodes: int, percentile: float):
        """
        Enhanced training with warm-start and adaptive strategies.
        
        Args:
            all_rewards: List to store all episode rewards
            num_epochs: Number of training epochs
            num_episodes: Number of episodes per epoch
            percentile: Base percentile for elite selection
        """
        
        # Store base percentile for adaptive adjustment
        self.base_percentile = percentile
        
        # Warm-start phase (if enabled)
        if self.use_warm_start:
            # Use multiprocessing version for efficiency
            if self.use_mp:
                warm_data = self.warm_start_with_multiprocessing()
            else:
                warm_data = self.warm_start_exploration()
            
            # Use warm-start data to initialize the policy
            warm_elite_obs, warm_elite_actions = self.select_elite_episodes(
                warm_data, percentile=50  # Be more inclusive for warm-start
            )
            
            if len(warm_elite_obs) > 0:
                print("\nTraining on warm-start elite episodes...")
                self.learn(warm_elite_obs, warm_elite_actions)
                
                # Add warm-start rewards to tracking
                warm_rewards = [ep[2] for ep in warm_data]
                all_rewards.extend(warm_rewards)
            else:
                print("Warning: No elite data from warm-start phase")
        
        # Continue with standard training loop
        print(f"\n{'='*60}")
        print(f"Starting main training loop")
        print(f"{'='*60}\n")
        
        should_add_noise = False

        pbar = tqdm(range(num_epochs), desc="Training", postfix={"mean_reward": 0})
        
        for e in pbar:
            # Collect data from multiple episodes
            episodes_data = []
            
            if self.use_mp:
                # Use multiprocessing for Atari environments
                policy_state_dict = self.state_dict()
                
                # Create tasks for multiprocessing
                tasks = []
                for i in range(num_episodes):
                    episode_seed = e * num_episodes + i
                    
                    task_args = (
                        policy_state_dict,
                        self.env.spec.id,
                        episode_seed,
                        self.hidden_dim,
                        self.is_atari,
                        False,  # random=False for training
                        should_add_noise # Should add noise to action selection
                    )
                    tasks.append(task_args)
                
                # Create a pool of workers and distribute the tasks
                with mp.Pool(processes=int(os.cpu_count() * self.cpu_usage)) as pool:
                    episodes_data = pool.map(evaluate_episode, tasks)
                
                # Extract rewards and add to all_rewards list
                episodic_rewards = [episode[2] for episode in episodes_data]
                all_rewards.extend(episodic_rewards)
                
            else:
                # Sequential processing (shouldn't happen for Atari, but kept for completeness)
                for _ in range(num_episodes):
                    history, total_reward = self.play_one_episode()
                    all_rewards.append(total_reward)
                    
                    episode_observations = [step[0] for step in history]
                    episode_actions = [step[1] for step in history]
                    
                    episodes_data.append((episode_observations, episode_actions, total_reward))
                
                episodic_rewards = [episode[2] for episode in episodes_data]
            
            # Calculate statistics
            mean_reward = np.mean(episodic_rewards)
            max_reward = max(episodic_rewards)
            min_reward = min(episodic_rewards)
            
            # Calculate adaptive percentile
            current_percentile = self.calculate_adaptive_percentile(episodic_rewards)

            reward_variance = np.var(episodic_rewards)
            if self.use_noise_injection and reward_variance < 0.1: # Threshold for "stagnation"
                print(f"Low reward variance ({reward_variance:.2f}), enabling action noise for next generation.")
                should_add_noise = True
            else:
                should_add_noise = False # Disable noise if performance is varied

            # Select elite with enhanced selection
            elite_obs, elite_actions = self.select_elite_episodes(
                episodes_data=episodes_data,
                percentile=current_percentile
            )
            
            # Train if we have elite data
            if len(elite_obs) > 0:
                self.learn(elite_obs, elite_actions)
                pbar.set_postfix_str(f"mean: {mean_reward:.2f}, "
                                    f"range: [{min_reward:.1f}, {max_reward:.1f}], "
                                    f"pct: {current_percentile}%"
                                    )
            else:
                pbar.set_postfix_str(f"mean: {mean_reward:.2f}, NO ELITE DATA")
            
            # Check for early termination
            if mean_reward > self.solved_threshold:
                pbar.close()
                print(f"\n{'='*60}")
                print(f"🎉 Environment solved! 🎉")
                print(f"Terminated after {e+1} epochs with mean reward {mean_reward:.2f}")
                print(f"{'='*60}")
                break

