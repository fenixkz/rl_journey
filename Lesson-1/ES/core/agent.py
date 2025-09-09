import multiprocessing as mp
import os
import sys
from copy import copy
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from core.policy import CNNNetwork, FCNetwork
from tqdm import tqdm

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../../")
sys.path.append(os.path.abspath(parent_path))

from common.base_agent import BaseAgent  # noqa: E402
from common.utils.atari_utils import clip_reward, get_atari_env  # noqa: E402


@dataclass
class MutantEvaluationParams:
    """Parameters for evaluating a single mutant in the ES algorithm."""

    mutant_weights: torch.Tensor
    env_id: str
    seed: int
    hidden_dim: int
    is_atari: bool
    use_vbn: bool
    action_noise: float
    vbn_stats: Optional[dict] = None


def evaluate_mutant(params: MutantEvaluationParams) -> float:
    """
    Asynchronous function that is using multiprocessing. Each offspring is using this function.
    Initializes an agent and environment, sets the weights for one mutant,
    plays an episode, and returns the reward.

    Args:
        params: MutantEvaluationParams containing all necessary parameters for evaluation

    Returns:
        float: Total reward obtained by the mutant in one episode
    """

    # Recreate the policy network structure
    agent = ESAgent(
        params.env_id,
        solved_threshold=9999,
        is_atari=params.is_atari,
        hidden_dim=params.hidden_dim,
        use_vbn=params.use_vbn,
        seed=params.seed,
    )

    # Set VBN statistics directly if using VBN, just copy-paste
    if params.use_vbn and params.vbn_stats is not None:
        agent.policy.set_vbn_stats(params.vbn_stats)

    # Set the weights for this specific mutant
    agent.set_weights(params.mutant_weights)

    # Play the episode and get the reward
    reward = agent.play_one_episode()

    return reward


class ESAgent(BaseAgent):
    """
    Evolution Strategies Agent with a policy represented by a neural network.
    The network accepts the state as the input and outputs action logits.
    Evolution happens by directly manipulating the network weights.
    Now includes Virtual Batch Normalization for improved exploration.
    """

    def __init__(
        self,
        env_id: str,
        solved_threshold: float,
        noise_std: float,
        is_atari: bool = False,
        hidden_dim: int = 64,
        seed: int = 224,
        learning_rate: float = 1e-3,
        cpu_usage: float = 0.5,
        use_vbn: bool = True,
        vbn_batch_size: int = 128,
        calculate_vbn_params: bool = False,
        l2_coeff: float = 0.005,
        normalization_mode: str = "default",
        action_noise: float = 0.1,
        evaluation_period: int = 100,
        evaluation_episodes: int = 50,
    ):
        """Initialize the Evolution Strategies agent with neural network policy.

        Creates an ES agent that evolves neural network weights through population-based
        optimization. Supports both classic control and Atari environments with optional
        Virtual Batch Normalization for improved exploration and training stability.

        Args:
            env_id: Gymnasium environment identifier (e.g., 'CartPole-v1')
            solved_threshold: Reward threshold considered as solved for early termination
            noise_std: Standard deviation for Gaussian noise perturbations during evolution
            is_atari: Whether the environment is an Atari game requiring special preprocessing
            hidden_dim: Number of hidden units in fully connected layers of the policy network
            seed: Random seed for reproducibility across training runs
            learning_rate: Learning rate for Adam optimizer used in weight updates
            cpu_usage: Fraction of available CPU cores to use for multiprocessing (0.0-1.0)
            use_vbn: Whether to enable Virtual Batch Normalization for better exploration
            vbn_batch_size: Size of reference batch for VBN statistics calculation
            calculate_vbn_params: Whether to immediately calculate VBN parameters during initialization
            l2_coeff: Coefficient for L2 regularization
        """
        # Create the environment using gymnasium
        env = gym.make(env_id) if not is_atari else get_atari_env(env_id)
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "Detected non-discrete action space, this class works only with discrete action space problems!"

        # Create the base agent
        super().__init__(env=env, solved_threshold=solved_threshold, seed=seed)

        # --- ES specific parameters
        self.noise_std = noise_std
        self.is_atari = is_atari
        self.cpu_usage = cpu_usage
        self.use_vbn = use_vbn
        self.vbn_batch_size = vbn_batch_size
        self.hidden_dim = hidden_dim
        self.normalization_mode = normalization_mode
        self.action_noise = action_noise
        # Overhead from creating agent and new env for simple Box2D envs is too much
        # easier to be sequential. But for Atari the opposite
        self.use_mp = is_atari
        # Store calculated statistics, not the full batch
        self.vbn_stats = None
        self.l2_coeff = l2_coeff
        self.evaluation_period = evaluation_period
        self.evaluation_episodes = evaluation_episodes

        if not is_atari and len(env.observation_space.shape) == 1:
            self.policy = FCNetwork(env.observation_space.shape[0], env.action_space.n, hidden_dim, use_vbn=use_vbn).to(
                self.device
            )
        else:
            self.policy = CNNNetwork(env.observation_space.shape, env.action_space.n, hidden_dim, use_vbn=use_vbn).to(
                self.device
            )

        # Optimizer for smarter weight updates
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Initialize Virtual Batch Normalization if enabled
        if self.use_vbn and calculate_vbn_params:
            self._initialize_vbn()

    def _initialize_vbn(self):
        """Initialize Virtual Batch Normalization by collecting reference batch statistics.

        Collects a diverse set of environment states by running random episodes and uses
        them to compute normalization statistics. This reference batch helps stabilize
        training and improves exploration by normalizing network inputs consistently
        across different policy parameter configurations during evolution.

        Note:
            The collected statistics are stored in self.vbn_stats for efficient transfer
            to multiprocessing workers without passing the entire reference batch.
        """
        reference_states = []

        # Collect reference batch from random episodes
        for _ in range(self.vbn_batch_size):
            state, _ = self.env.reset()
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
        reference_states = reference_states[: self.vbn_batch_size]

        # Convert to tensor
        reference_batch = torch.tensor(np.array(reference_states), dtype=torch.float32).to(self.device)

        # Set reference batch in the policy network (calculates statistics once)
        self.policy.set_reference_batch(reference_batch)

        # Extract and store the calculated statistics for efficient passing to workers
        self.vbn_stats = self.policy.get_vbn_stats()

    def choose_action(self, state: np.ndarray) -> int:
        """Select an action using the current policy network in a deterministic manner.

        Feeds the environment state through the neural network policy to obtain action
        logits, then selects the action with highest probability (argmax). This deterministic
        approach is typical for Evolution Strategies where we evaluate each policy variant
        consistently without additional exploration noise.

        Args:
            state: Current environment observation as numpy array

        Returns:
            Selected discrete action index as integer
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits: torch.Tensor = self.policy.forward(state)
        noise = torch.randn_like(logits) * self.action_noise
        noisy_logits = logits + noise
        return torch.argmax(noisy_logits, dim=1).item()

    def get_weights(self):
        """Extract all trainable parameters from the policy network as a flattened vector.

        Concatenates all parameter tensors (weights and biases) from the neural network
        into a single 1D tensor. This flattened representation is essential for Evolution
        Strategies operations like adding noise perturbations and computing parameter updates.

        Returns:
            torch.Tensor: Flattened vector containing all network parameters
        """
        return torch.cat([p.data.flatten() for p in self.policy.parameters()])

    def set_weights(self, weights_flat):
        """Update all policy network parameters from a flattened weight vector.

        Takes a 1D tensor containing all network parameters and reshapes/assigns them
        back to their original parameter tensor shapes in the neural network. This is
        the inverse operation of get_weights() and enables setting mutated parameters
        during Evolution Strategies evaluation.

        Args:
            weights_flat: 1D tensor containing flattened network parameters to be assigned
        """
        offset = 0
        for param in self.policy.parameters():
            param_shape = param.data.shape
            num_elements = param.data.numel()
            # Reshape the flat vector segment and assign it to the parameter
            param.data.copy_(weights_flat[offset : offset + num_elements].reshape(param_shape))
            offset += num_elements

    def save(self, save_path: str):
        """Save the current policy network parameters to disk for later use.

        Creates the specified directory if it doesn't exist and saves the complete
        state dictionary of the neural network policy. This includes all learned
        weights, biases, and any other trainable parameters that define the current
        policy behavior after evolution.

        Args:
            save_path: Directory path where the policy file will be saved
        """
        os.makedirs(save_path, exist_ok=True)
        print("--- Saving Policy ---")
        torch.save(self.policy.state_dict(), f"{save_path}/policy.pth")
        print(f"Policy saved in {save_path}")

    def load(self, load_path: str):
        """Load previously saved policy network parameters from disk.

        Attempts to restore a trained policy from the specified path. If the policy
        file is not found, the agent continues with randomly initialized parameters.
        This method enables resuming training or deploying a pre-trained policy for
        evaluation without retraining from scratch.

        Args:
            load_path: Directory path where the policy file is expected to be located
        """
        try:
            self.policy.load_state_dict(torch.load(f"{load_path}/policy.pth"))
            print(f"Policy loaded from {load_path}")
        except FileNotFoundError:
            print(f"Policy file not found in {load_path}. Using untrained policy.")

    def play_one_episode(self, seed: Optional[int] = None):
        """Execute a complete episode using the current policy and return total reward.

        Runs the agent through one full episode in the environment, making deterministic
        action selections based on the current policy network. Handles environment resets,
        Atari-specific preprocessing (auto-fire, reward clipping), and accumulates rewards
        until episode termination. Used for fitness evaluation during evolution.

        Returns:
            float: Total accumulated reward from the complete episode
        """
        if seed is not None:
            state, _ = self.env.reset()
        else:
            state, _ = self.env.reset(seed=seed)
        done = False
        total_reward = 0
        while not done:
            action = self.choose_action(state=state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            if self.is_atari:
                reward = clip_reward(reward=reward)
            state = next_state
            total_reward += reward
        return total_reward

    def normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        # Three normalization modes

        # 1. Default: scale all rewards to have zero mean and 1 variance
        if self.normalization_mode == "default":
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # 2. One: scale rewards by dividing by the maximum to have maximum reward of one
        if self.normalization_mode == "one":
            rewards /= rewards.max() + 1e-8
            return rewards

        # 3. Rank: rank based normalization per OpenAI implementation
        # Find a rank for each reward and then clamps them to [-0.5, 0.5] range
        if self.normalization_mode == "rank":
            ranks = np.empty(rewards.size, dtype=int)
            ranks[rewards.argsort()] = np.arange(rewards.size)
            rewards = ranks.astype(np.float32) / (rewards.size - 1) - 0.5

        return rewards

    def train(self, num_epochs: int, population_size: int):
        """
        Execute the main Evolution Strategies training algorithm with antithetic sampling.

        Implements the core ES training loop where a population of policy variants is
        generated by adding Gaussian noise to current parameters, evaluated in parallel
        or sequentially, and used to compute parameter updates. Uses antithetic sampling
        (±ε pairs) to reduce variance and Adam optimizer for adaptive learning rates.
        Supports multiprocessing for efficient evaluation and early stopping when solved.

        The algorithm follows these steps per generation:
        1. Sample noise vectors and create ±ε parameter perturbations
        2. Evaluate fitness (episode rewards) for each variant
        3. Compute weighted parameter updates based on fitness differences
        4. Apply updates using Adam optimizer with gradient ascent
        5. Check for convergence and early termination

        Args:
            all_rewards: List to accumulate all episode rewards for plotting and analysis
            num_epochs: Maximum number of generations to run the evolution process
            population_size: Number of policy variants to evaluate each generation (must be even)
        """
        num_weights = self.get_weights().numel()
        mean_eval_rewards = 0
        pbar = tqdm(range(num_epochs), desc="Evolving", postfix={"mean_reward": 0})

        for generation in pbar:
            # Step 1. Get DNA of the parent
            # DNA means the weights of the central agent
            central_weights = self.get_weights()

            # Step 2: Create a population of offsprings

            # 1. Generate HALF the number of noise vectors in both directions +ε and -ε
            # This is antithetic noise, by doing + and - noise, we create a zero random bias
            num_pairs = population_size // 2
            noise_vectors = [torch.randn(num_weights, device=self.device) for _ in range(num_pairs)]

            # Step 3: Let the offsprings play and calculate their fitness
            # Either using multiprocessing or sequential approach
            if self.use_mp:
                # 1. Create tasks for BOTH the positive and negative perturbations
                tasks = []
                vbn_stats_to_pass = self.vbn_stats if self.use_vbn else None

                for i, noise in enumerate(noise_vectors):
                    # Each pair should compete in the same initial conditions
                    # But different pairs should be exposed to different conditions for better generalization
                    pair_seed = generation * population_size + i

                    base_task_params = MutantEvaluationParams(
                        env_id=self.env.spec.id,
                        seed=pair_seed,
                        hidden_dim=self.hidden_dim,
                        is_atari=self.is_atari,
                        use_vbn=self.use_vbn,
                        vbn_stats=vbn_stats_to_pass,
                        action_noise=self.action_noise,
                    )
                    # Create a positive mutant "DNA" and its corresponding task
                    task_params_plus = copy(base_task_params)
                    task_params_plus.mutant_weights = central_weights + noise * self.noise_std
                    tasks.append(task_params_plus)

                    # Create the a negative mutant "DNA" and its corresponding task
                    task_params_minus = copy(base_task_params)
                    task_params_minus.mutant_weights = central_weights - noise * self.noise_std
                    tasks.append(task_params_minus)

                # Create a pool of workers and distribute the tasks, use some part of total cpu power
                with mp.Pool(processes=int(os.cpu_count() * self.cpu_usage)) as pool:
                    # pool.map calls evaluate_mutant() for each item in 'tasks'
                    # it returns a list of rewards for each task
                    rewards = pool.map(evaluate_mutant, tasks)

                # Populate the external list for plotting purposes
                self.train_rewards.extend(rewards)
                # Cast to np.array for better calculations
                rewards = np.array(rewards)
            else:
                # A list to collect all rewards
                sequential_rewards = []
                for i, noise in enumerate(noise_vectors):
                    # Same logic as in mp branch
                    pair_seed = generation * population_size + i

                    # Evaluate a positive mutant "DNA"
                    self.set_weights(central_weights + self.noise_std * noise)
                    reward_plus = self.play_one_episode(seed=pair_seed)
                    sequential_rewards.append(reward_plus)

                    # Evaluate a negative mutant "DNA"
                    self.set_weights(central_weights - self.noise_std * noise)
                    reward_minus = self.play_one_episode(seed=pair_seed)
                    sequential_rewards.append(reward_minus)

                # Populate the external list for plotting purposes
                self.train_rewards.extend(sequential_rewards)
                # Cast to np.array for better calculations
                rewards = np.array(sequential_rewards)
                # Set the central weights back to its original value
                self.set_weights(central_weights)

            # Track performance
            mean_reward = np.mean(rewards)

            # Step 4: Natural Selection and Generational Update
            # This is the core update rule: θ_new = θ_old + learning_rate * Σ(R_i * ε_i)

            # 1. A common trick: normalize rewards to prevent extreme updates
            rewards = self.normalize_rewards(rewards)

            # 2. Calculate the direction to move each weight in central agent
            update_direction = torch.zeros(num_weights, device=self.device)
            for i in range(num_pairs):
                # Get the rewards for the +ε
                reward_plus = rewards[2 * i]
                # and -ε pair
                reward_minus = rewards[2 * i + 1]

                """
                Combine their influence on the update direction.
                We use (reward_plus - reward_minus) to determine which direction was better.

                Case 1: reward_plus > reward_minus
                The result is a positive number. Multiplying the positive noise vector `noise_vectors[i]`
                by this positive number means we move the update in the direction of `+ε`.
                This makes sense, as the positive perturbation led to a higher reward.

                Case 2: reward_minus > reward_plus
                The result is a negative number. Multiplying `noise_vectors[i]` by this negative
                number is equivalent to adding in the direction of `-ε`.
                This also makes sense, as the negative perturbation was more successful.

                The magnitude of the difference also scales the update, so pairs with a larger
                performance gap have more influence on the final direction.
                """
                update_direction += (reward_plus - reward_minus) * noise_vectors[i]

            # 3. Evolve the parent's DNA by taking a small step in the successful direction

            # The (population_size * noise_std) term is a standard normalization part of the ES gradient estimator
            update_step = (1 / (population_size * self.noise_std)) * update_direction

            # Manually assign the calculated gradients to the policy's parameters
            # Using Adam optimizer is more effective, because of its internal momentum and adaptive lr calculations
            self.optimizer.zero_grad()
            offset = 0
            for layer in self.policy.parameters():
                # Get the total number of parameters in this specific layer
                num_elements = layer.numel()
                # Find corresponding update direction for this set of params
                reward_gradient = update_step[offset : offset + num_elements].reshape(layer.shape)
                # Add L2 penalty term for regularization
                l2_penalty = self.l2_coeff * layer.data
                # We want to maximize, thus we need to minimize the negative of it
                layer.grad = -(reward_gradient - l2_penalty).to(self.device)
                offset += num_elements

            # Tell Adam to perform its update step
            self.optimizer.step()

            if generation % self.evaluation_period == 0:
                eval_rewards_ = [self.play_one_episode() for _ in range(self.evaluation_episodes)]
                mean_eval_rewards = np.mean(eval_rewards_)
                # Add to the external list
                self.val_rewards.append(mean_eval_rewards)
            pbar.set_postfix_str(f"Train mean: {mean_reward:.2f} Eval mean: {mean_eval_rewards:.2f}")

            # Check for early termination
            if mean_eval_rewards > self.solved_threshold:
                pbar.close()
                print(f"Terminated after {generation+1} steps with mean reward {mean_reward:.2f}")
                break
