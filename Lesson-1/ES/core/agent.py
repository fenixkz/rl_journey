import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import time
import os
import sys
from tqdm import tqdm
import random
from typing import List
import multiprocessing as mp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.atari_utils import get_atari_env, clip_reward, auto_fire
from collections import deque

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

class VirtualBatchNorm1d(nn.Module):
    """
    Virtual Batch Normalization for 1D inputs (fully connected layers).
    Uses a reference batch to compute normalization statistics, making the network
    more sensitive to parameter perturbations during evolution.
    """
    def __init__(self, num_features, eps=1e-5):
        super(VirtualBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Register buffers for reference batch statistics
        self.register_buffer('ref_mean', torch.zeros(num_features))
        self.register_buffer('ref_var', torch.ones(num_features))
        self.register_buffer('initialized', torch.tensor(False))
    
    def set_reference_stats(self, mean, var):
        """Set statistics from reference batch"""
        with torch.no_grad():
            self.ref_mean.copy_(mean)
            self.ref_var.copy_(var)
            self.initialized.fill_(True)
    
    def forward(self, x):
        if not self.initialized:
            # If not initialized, use current batch statistics (fallback)
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
        else:
            # Use ONLY reference batch statistics for normalization
            batch_mean = self.ref_mean
            batch_var = self.ref_var
        
        # Normalize using reference statistics
        x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        
        # Scale and shift
        return self.gamma * x_normalized + self.beta

class VirtualBatchNorm2d(nn.Module):
    """
    Virtual Batch Normalization for 2D inputs (convolutional layers).
    Uses a reference batch to compute normalization statistics.
    """
    def __init__(self, num_features, eps=1e-5):
        super(VirtualBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Register buffers for reference batch statistics
        self.register_buffer('ref_mean', torch.zeros(num_features))
        self.register_buffer('ref_var', torch.ones(num_features))
        self.register_buffer('initialized', torch.tensor(False))
    
    def set_reference_stats(self, mean, var):
        """Set statistics from reference batch"""
        with torch.no_grad():
            self.ref_mean.copy_(mean)
            self.ref_var.copy_(var)
            self.initialized.fill_(True)
    
    def forward(self, x):
        if not self.initialized:
            # If not initialized, use current batch statistics (fallback)
            batch_mean = x.mean(dim=[0, 2, 3])
            batch_var = x.var(dim=[0, 2, 3], unbiased=False)
        else:
            # Use ONLY reference batch statistics for normalization
            batch_mean = self.ref_mean
            batch_var = self.ref_var
        
        # Reshape for broadcasting
        batch_mean = batch_mean.view(1, -1, 1, 1)
        batch_var = batch_var.view(1, -1, 1, 1)
        
        # Normalize using reference statistics
        x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        
        # Scale and shift (reshape gamma and beta for broadcasting)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        
        return gamma * x_normalized + beta

class FCNetwork(nn.Module):
    '''
    A fully-connected neural network. This is our policy, π(θ).
    Given the state, returns action logits for Evolution Strategies.
    Now includes Virtual Batch Normalization for better exploration.
    '''
    
    def __init__(self, obs_space, action_space, hidden_space: int = 64, use_vbn: bool = True):
        super(FCNetwork, self).__init__()
        self.use_vbn = use_vbn
        
        if use_vbn:
            self.vbn1 = VirtualBatchNorm1d(hidden_space)
            self.vbn2 = VirtualBatchNorm1d(hidden_space)
            
            self.network = nn.Sequential(
                nn.Linear(obs_space, hidden_space),
                self.vbn1,
                nn.ReLU(),
                nn.Linear(hidden_space, hidden_space),
                self.vbn2,
                nn.ReLU(),
                nn.Linear(hidden_space, action_space)
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(obs_space, hidden_space),
                nn.ReLU(),
                nn.Linear(hidden_space, hidden_space),
                nn.ReLU(),
                nn.Linear(hidden_space, action_space)
            )
        self._init_weights()
    
    def set_reference_batch(self, reference_states):
        """Process reference batch through network to set VBN statistics using hooks"""
        if not self.use_vbn:
            return
        
        # Dictionary to store intermediate activations
        activations = {}
        
        # Define hook function to capture activations
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        # Register hooks on VBN layers
        hooks = []
        vbn_layers = {'vbn1': self.vbn1, 'vbn2': self.vbn2}
        for name, layer in vbn_layers.items():
            handle = layer.register_forward_hook(get_activation(name))
            hooks.append(handle)
        
        # Forward pass with reference batch
        with torch.no_grad():
            _ = self.forward(reference_states)
        
        # Set statistics for each VBN layer
        for name, layer in vbn_layers.items():
            if name in activations:
                activation = activations[name]
                mean = activation.mean(dim=0)
                var = activation.var(dim=0, unbiased=False)
                layer.set_reference_stats(mean, var)
        
        # Remove hooks
        for handle in hooks:
            handle.remove()
    
    def get_vbn_stats(self):
        """Extract VBN statistics to pass to workers"""
        if not self.use_vbn:
            return None
        
        stats = {}
        vbn_layers = {'vbn1': self.vbn1, 'vbn2': self.vbn2}
        for name, layer in vbn_layers.items():
            stats[name] = {
                'mean': layer.ref_mean.cpu(),
                'var': layer.ref_var.cpu()
            }
        return stats
    
    def set_vbn_stats(self, stats):
        """Set VBN statistics directly (for workers)"""
        if not self.use_vbn or stats is None:
            return
        
        vbn_layers = {'vbn1': self.vbn1, 'vbn2': self.vbn2}
        for name, layer in vbn_layers.items():
            if name in stats:
                layer.set_reference_stats(
                    stats[name]['mean'].to(layer.ref_mean.device),
                    stats[name]['var'].to(layer.ref_var.device)
                )
        
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
    This is our policy, π(θ).
    Given the state, returns action logits for Evolution Strategies.
    Now includes Virtual Batch Normalization for better exploration.
    '''
    def __init__(self, obs_space, action_space, hidden_space: int = 64, use_vbn: bool = True):
        super(CNNNetwork, self).__init__()
        self.use_vbn = use_vbn
        C, H, W = obs_space
        
        if use_vbn:
            self.vbn1 = VirtualBatchNorm2d(32)
            self.vbn2 = VirtualBatchNorm2d(64)
            self.vbn3 = VirtualBatchNorm2d(64)
            self.vbn4 = VirtualBatchNorm1d(hidden_space)
            
            self.backbone = nn.Sequential(
                nn.Conv2d(C, 32, kernel_size=8, stride=4),
                self.vbn1,
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                self.vbn2,
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                self.vbn3,
                nn.ReLU()
            )
        else:
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
            
        if use_vbn:
            self.regressor = nn.Sequential(
                nn.Linear(feature_size, hidden_space),
                self.vbn4,
                nn.ReLU(),
                nn.Linear(hidden_space, action_space)
            )
        else:
            self.regressor = nn.Sequential(
                nn.Linear(feature_size, hidden_space),
                nn.ReLU(),
                nn.Linear(hidden_space, action_space)
            )
        self._init_weights()
    
    def set_reference_batch(self, reference_states):
        """Process reference batch through network to set VBN statistics using hooks"""
        if not self.use_vbn:
            return
        
        # Dictionary to store intermediate activations
        activations = {}
        
        # Define hook function to capture activations
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        # Register hooks on VBN layers
        hooks = []
        vbn_layers = {
            'vbn1': self.vbn1,
            'vbn2': self.vbn2,
            'vbn3': self.vbn3,
            'vbn4': self.vbn4
        }
        
        for name, layer in vbn_layers.items():
            handle = layer.register_forward_hook(get_activation(name))
            hooks.append(handle)
        
        # Forward pass with reference batch
        with torch.no_grad():
            _ = self.forward(reference_states)
        
        # Set statistics for each VBN layer
        for name, layer in vbn_layers.items():
            if name in activations:
                activation = activations[name]
                # For Conv2d layers (vbn1-3)
                if name in ['vbn1', 'vbn2', 'vbn3']:
                    mean = activation.mean(dim=[0, 2, 3])
                    var = activation.var(dim=[0, 2, 3], unbiased=False)
                # For Linear layer (vbn4)
                else:
                    mean = activation.mean(dim=0)
                    var = activation.var(dim=0, unbiased=False)
                layer.set_reference_stats(mean, var)
        
        # Remove hooks
        for handle in hooks:
            handle.remove()
    
    def get_vbn_stats(self):
        """Extract VBN statistics to pass to workers"""
        if not self.use_vbn:
            return None
        
        stats = {}
        vbn_layers = {
            'vbn1': self.vbn1,
            'vbn2': self.vbn2,
            'vbn3': self.vbn3,
            'vbn4': self.vbn4
        }
        for name, layer in vbn_layers.items():
            stats[name] = {
                'mean': layer.ref_mean.cpu(),
                'var': layer.ref_var.cpu()
            }
        return stats
    
    def set_vbn_stats(self, stats):
        """Set VBN statistics directly (for workers)"""
        if not self.use_vbn or stats is None:
            return
        
        vbn_layers = {
            'vbn1': self.vbn1,
            'vbn2': self.vbn2,
            'vbn3': self.vbn3,
            'vbn4': self.vbn4
        }
        for name, layer in vbn_layers.items():
            if name in stats:
                layer.set_reference_stats(
                    stats[name]['mean'].to(layer.ref_mean.device),
                    stats[name]['var'].to(layer.ref_var.device)
                )

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


def evaluate_mutant(args):
    """
    Asynchronous function that is using multiprocessing. Each offspring is using this function.
    Initializes an agent and environment, sets the weights for one mutant,
    plays an episode, and returns the reward.
    """
    # Unpack the arguments
    central_weights_flat, noise_vector, noise_std, env_id, seed, hidden_dim, is_atari, use_vbn, vbn_stats = args
    
    # Each process needs its own environment and agent instance
    env = gym.make(env_id) if not is_atari else get_atari_env(env_id)
    # Use a different seed for each worker's env to ensure diverse episodes
    env.reset(seed=seed)
    env.action_space.seed(seed)

    # Recreate the policy network structure
    agent = ESAgent(env, solved_threshold=9999, is_atari=is_atari, hidden_dim=hidden_dim, use_vbn=use_vbn)
    
    # Set VBN statistics directly if using VBN (much more efficient than passing full batch)
    if use_vbn and vbn_stats is not None:
        agent.policy.set_vbn_stats(vbn_stats)
    
    # Set the weights for this specific mutant
    mutant_weights = central_weights_flat + noise_std * noise_vector
    agent.set_weights(mutant_weights)
    
    # Play the episode and get the reward
    reward = agent.play_one_episode()
    
    env.close()
    return reward


class ESAgent(nn.Module):
    '''
    Evolution Strategies Agent with a policy represented by a neural network.
    The network accepts the state as the input and outputs action logits.
    Evolution happens by directly manipulating the network weights.
    Now includes Virtual Batch Normalization for improved exploration.
    '''
    def __init__(self,
                 env: gym.Env,
                 solved_threshold: float,
                 is_atari: bool = False,
                 hidden_dim: int = 64,
                 seed: int = 224,
                 learning_rate: float = 1e-3,
                 cpu_usage: float = 0.5,
                 use_vbn: bool = True,
                 vbn_batch_size: int = 128):
        assert(isinstance(env.action_space, gym.spaces.Discrete)), "Detected non-discrete action space, this class works only with discrete action space problems!"
        set_seed(seed)
        super(ESAgent, self).__init__()

        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.solved_threshold = solved_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_atari = is_atari
        self.hidden_dim = hidden_dim
        self.use_mp = is_atari # Overhead from creating agent and new env for simple Box2D envs is too much, easier to be sequential. But for Atari the opposite
        self.lr = learning_rate
        self.cpu_usage = cpu_usage
        self.use_vbn = use_vbn
        self.vbn_batch_size = vbn_batch_size
        self.vbn_stats = None  # Store calculated statistics, not the full batch
        
        # Add a deque to track recent performance for adaptive noise
        self.performance_history = deque(maxlen=10) # Track last 10 generations
        if not is_atari:
            # print(f"Detected a classic (vector) environment, observation space shape: {self.observation_space.shape}, using Fully Connected (FC) Network")
            self.policy = FCNetwork(self.observation_space.shape[0], self.action_space.n, hidden_dim, use_vbn=use_vbn).to(self.device)
        else:
            # print(f"Detected an Atari environment, observation space shape: {self.observation_space.shape}, using Convolutional Neural Network (CNN)")
            self.policy = CNNNetwork(self.observation_space.shape, self.action_space.n, hidden_dim, use_vbn=use_vbn).to(self.device)
        
        # Optimizer for smarter weight updates
        self.optimizer = torch.optim.Adam(
                                            self.policy.parameters(),
                                            lr=self.lr,
                                            weight_decay=1e-4  # <-- Some regularization
                                        )
        # First reset for reproducing same results
        env.reset(seed=seed)
        env.action_space.seed(seed)
        
        # Initialize Virtual Batch Normalization if enabled
        if self.use_vbn:
            self._initialize_vbn()
    
    def _initialize_vbn(self):
        """Collect a reference batch and initialize VBN statistics"""
        reference_states = []
        
        # Collect reference batch from random episodes
        for _ in range(self.vbn_batch_size):
            state, _ = self.env.reset()
            if self.is_atari:
                state = auto_fire(self.env)
            reference_states.append(state)
            
            # Take a few random steps to get diverse states
            for _ in range(np.random.randint(1, 10)):
                action = self.env.action_space.sample()
                next_state, _, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    break
                state = next_state
                reference_states.append(state)
                
                if len(reference_states) >= self.vbn_batch_size:
                    break
            
            if len(reference_states) >= self.vbn_batch_size:
                break
        
        # Trim to exact batch size
        reference_states = reference_states[:self.vbn_batch_size]
        
        # Convert to tensor
        reference_batch = torch.tensor(np.array(reference_states), dtype=torch.float32).to(self.device)
        
        # Set reference batch in the policy network (calculates statistics once)
        self.policy.set_reference_batch(reference_batch)
        
        # Extract and store the calculated statistics for efficient passing to workers
        self.vbn_stats = self.policy.get_vbn_stats()
        

    def choose_action(self, state: np.ndarray) -> int:
        """Chooses an action based on the policy's output logits."""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.policy.forward(state_t)
        # For ES, we typically use deterministic actions (argmax)
        return logits.argmax(dim=-1).item()

    def get_weights(self):
        """Helper function to get network weights as a single flat vector."""
        return torch.cat([p.data.flatten() for p in self.policy.parameters()])

    def set_weights(self, weights_flat):
        """Helper function to set network weights from a single flat vector."""
        offset = 0
        for param in self.policy.parameters():
            param_shape = param.data.shape
            num_elements = param.data.numel()
            # Reshape the flat vector segment and assign it to the parameter
            param.data.copy_(weights_flat[offset:offset + num_elements].reshape(param_shape))
            offset += num_elements

    def save_policy(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        print("--- Saving Policy ---")
        torch.save(self.state_dict(), f"{save_path}/policy.pth")
        print(f"Policy saved in {save_path}")

    def load_policy(self, save_path: str):
        try:
            self.load_state_dict(
                torch.load(f"{save_path}/policy.pth")
            )
            print(f"Policy loaded from {save_path}")
        except FileNotFoundError:
            print(f"Policy file not found in {save_path}. Using untrained policy.")

    def play_one_episode(self):
        """Plays one episode using the provided agent and returns the total reward."""
        state, _ = self.env.reset()
        if self.is_atari: state = auto_fire(self.env)
        done = False
        total_reward = 0
        while not done:
            action = self.choose_action(state=state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            if self.is_atari: reward = clip_reward(reward=reward)
            state = next_state
            total_reward += reward
        return total_reward

    def train(self, all_rewards: List, num_epochs: int, population_size: int, noise_std: float):
        """
        Evolution Strategies training loop.
        """
        num_weights = self.get_weights().numel()
        # Set the current noise scale
        current_noise_std = noise_std

        pbar = tqdm(range(num_epochs), desc="Evolving", postfix={"mean_reward": 0})

        for generation in pbar:
            # Step 1. Get DNA of the parent
            # DNA means the weights of the central agent
            central_weights = self.get_weights()
            

            # Step 2: Create a Population of "Mutants"
            # 1. Generate HALF the number of noise vectors in both directions +ε and -ε
            # We want + and - to create perfectly anti-correlated pairs of perturbations, such that the total sum of all perturbations is zero and there is no any bias
            num_pairs = population_size // 2
            noise_vectors = [torch.randn(num_weights, device=self.device) for _ in range(num_pairs)]

            if self.use_mp:
                # 2. Create tasks for BOTH the positive and negative perturbations, this is needed for multiprocessing
                tasks = []
                # Pass only the small statistics tensors to workers (not the full batch)
                vbn_stats_to_pass = self.vbn_stats if self.use_vbn else None
                
                for i, noise in enumerate(noise_vectors):
                    base_seed = generation * population_size + i

                    task_args = (
                        central_weights.cpu(),
                        noise.cpu(),  # <-- Use the positive noise
                        current_noise_std,
                        self.env.spec.id,
                        base_seed,
                        self.hidden_dim,
                        self.is_atari,
                        self.use_vbn,
                        vbn_stats_to_pass
                    )
                    # Add the +ε task
                    tasks.append(task_args)
                    
                    # Create the arguments for the -ε task
                    task_args_minus = (
                        central_weights.cpu(),
                        -noise.cpu(),        # <-- Use the negative noise
                        current_noise_std,
                        self.env.spec.id,
                        base_seed + num_pairs, # Use a different seed
                        self.hidden_dim,
                        self.is_atari,
                        self.use_vbn,
                        vbn_stats_to_pass
                    )
                    # Add the -ε task
                    tasks.append(task_args_minus)

                # Step 3: Distribute offsprings
                # Create a pool of workers and distribute the tasks
                # This automatically uses percentage of all available CPU cores
                with mp.Pool(processes=int(os.cpu_count() * self.cpu_usage)) as pool:
                    # pool.map calls evaluate_mutant for each item in 'tasks' and collects the results
                    # map also returns a list of rewards
                    rewards = pool.map(evaluate_mutant, tasks)
                
                # Populate the external list for plotting purposes
                all_rewards.extend(rewards)
                # Cast to np.array for better calculations
                rewards = np.array(rewards)
            else:
                sequential_rewards = []
                # Evaluate the Population's "Fitness" using antithetic pairs
                for noise in noise_vectors:
                    # Evaluate the +ε version
                    self.set_weights(central_weights + current_noise_std * noise)
                    reward_plus = self.play_one_episode()
                    sequential_rewards.append(reward_plus)

                    # Evaluate the -ε version
                    self.set_weights(central_weights - current_noise_std * noise)
                    reward_minus = self.play_one_episode()
                    sequential_rewards.append(reward_minus)
                # Populate the external list for plotting purposes
                all_rewards.extend(sequential_rewards)
                # Cast to np.array for better calculations
                rewards = np.array(sequential_rewards)
            
            # Track performance
            mean_reward = np.mean(rewards)
            pbar.set_postfix_str(f"mean_reward: {mean_reward:.2f}")
            
            # 1. Update performance history
            self.performance_history.append(mean_reward)

            # 2. Check for stagnation
            is_stuck = False
            if len(self.performance_history) == self.performance_history.maxlen:
                # Check if improvement over the last 10 generations is minimal
                improvement = max(self.performance_history) - min(self.performance_history)
                if abs(improvement) < 0.5: # If score hasn't improved by at least 0.5
                    is_stuck = True
            
            # 3. Adapt the noise standard deviation
            if is_stuck:
                # If stuck, increase noise to explore more widely
                current_noise_std *= 1.1
                pbar.set_postfix_str(f"mean_reward: {mean_reward:.2f} (Stuck! ↑ noise: {current_noise_std:.3f})")
            else:
                # If improving, decrease noise for finer tuning
                current_noise_std *= 0.5
                pbar.set_postfix_str(f"mean_reward: {mean_reward:.2f} (Improving ↓ noise: {current_noise_std:.3f})")

            # Add bounds to prevent noise from exploding or vanishing
            current_noise_std = max(noise_std, min(0.5, current_noise_std))

            # Step 4: Natural Selection and Generational Update
            # This is the core update rule: θ_new = θ_old + learning_rate * Σ(R_i * ε_i)
            
            # A common trick: normalize rewards to prevent extreme updates
            normalized_rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
            
            # Calculate the weighted sum of mutations
            update_direction = torch.zeros(num_weights, device=self.device)
            for i in range(num_pairs):
                # Get the rewards for the +ε 
                reward_plus = normalized_rewards[2 * i]
                # and -ε pair
                reward_minus = normalized_rewards[2 * i + 1]
                
                # Combine their influence on the update direction
                update_direction += (reward_plus - reward_minus) * noise_vectors[i]
                
            # Evolve the parent's DNA by taking a small step in the successful direction
            # The (population_size * noise_std) term is a standard part of the ES gradient estimator
            # We want to MAXIMIZE reward, so we move along the gradient.
            # Adam MINIMIZES loss, so it moves in the NEGATIVE gradient direction.
            # Therefore, we set .grad to the NEGATIVE of our estimated gradient.
            update_step = (1 / (population_size * current_noise_std)) * update_direction
            # Manually assign the calculated gradients to the policy's parameters
            self.optimizer.zero_grad()
            offset = 0
            for param in self.policy.parameters():
                num_elements = param.numel()
                # Reshape the gradient segment and assign it
                param.grad = -update_step[offset:offset + num_elements].reshape(param.shape).to(self.device)
                offset += num_elements

            # Tell Adam to perform its update step
            self.optimizer.step()

            # Check for early termination
            if mean_reward > self.solved_threshold:
                pbar.close()
                print(f"Terminated after {generation+1} steps with mean reward {mean_reward:.2f}")
                break
            