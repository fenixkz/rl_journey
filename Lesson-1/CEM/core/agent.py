import os
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.policy import FCNetwork
from torch.distributions import Categorical
from tqdm import tqdm

from common.base_agent import BaseAgent


class CEMAgent(BaseAgent):
    """
    This is our Agent with a policy represented by a neural network.
    """

    def __init__(
        self,
        env_id: str,
        solved_threshold: float,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        seed: int = 224,
    ):
        """
        Initialize the Cross-Entropy Method (CEM) agent.

        Args:
            env_id (str): Environment identifier for gymnasium environment.
            solved_threshold (float): Mean reward threshold to consider the environment solved.
            hidden_dim (int, optional): Number of hidden units in the neural network. Defaults to 64.
            lr (float, optional): Learning rate for the Adam optimizer. Defaults to 1e-3.
            seed (int, optional): Random seed for reproducibility. Defaults to 224.

        Raises:
            AssertionError: If the environment has non-discrete action space.
        """

        # Create the environment using gymnasium
        env = gym.make(env_id)
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "Detected non-discrete action space, this class works only with discrete action space problems!"
        assert (
            len(env.observation_space.shape) > 1
        ), "Detected non-vector environment, Cross-Entropy Agent only works with Vector Envs"

        # Create the base agent
        super().__init__(env=env, solved_threshold=solved_threshold, seed=seed)

        # --- Create a policy, optimizer and cross-entropy loss ---
        self.policy = FCNetwork(env.observation_space.shape[0], env.action_space.n, hidden_dim, self.device).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

    def choose_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Choose an action based on the current observation.

        This method performs a forward pass through the policy network and either
        selects the action with highest probability (deterministic) or samples
        from the action probability distribution (stochastic).

        Args:
            observation (np.ndarray): Current state observation from the environment.
            deterministic (bool, optional): If True, selects the action with highest
                probability. If False, samples from the probability distribution.
                Defaults to False.

        Returns:
            np.ndarray: Selected action as a scalar integer.
        """
        # 1. Cast to tensor and add a batch dimension
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 2. Do the forward pass and get raw logits
        logits: torch.Tensor = self.policy.forward(observation)

        # 3. If determinstic, then just return the index of action corresponding to the maximum logit
        if deterministic:
            return logits.argmax(dim=-1).item()

        # 4. Convert raw logits into valid probabilities
        probs = F.softmax(logits, dim=-1)

        # 5. Create a torch distribution object
        dist = Categorical(probs)

        # 6. Sample an action from the distribution
        action = dist.sample()

        # 7. Return as scalar
        return action.item()

    def select_elite_episodes(self, episodes_data: List[Tuple[np.ndarray, int, float]], percentile: float):
        """
        Select elite episodes based on total episode rewards using percentile threshold.

        This method sorts episodes by their total rewards in descending order and
        selects the top performing episodes based on the given percentile. The elite
        episodes' states and actions are then extracted for policy training.

        Args:
            episodes_data (list): List of tuples containing (states, actions, total_reward)
                for each episode.
            percentile (float): Percentile threshold for selecting elite episodes.
                Higher values mean fewer elite episodes are selected.

        Returns:
            tuple: A tuple containing:
                - elite_obs (np.ndarray): Array of all observations from elite episodes.
                - elite_actions (np.ndarray): Array of all actions from elite episodes.
        """
        # Sort episodes by total reward
        episodes_data.sort(key=lambda x: x[2], reverse=True)

        # Calculate how many episodes to keep
        n_elite = int(len(episodes_data) * (100 - percentile) / 100)
        n_elite = max(1, n_elite)  # Keep at least 1 episode

        # Get elite episodes
        elite_episodes = episodes_data[:n_elite]

        # Extract all states and actions from elite episodes
        elite_obs = []
        elite_actions = []

        for states, actions, _ in elite_episodes:
            elite_obs.extend(states)
            elite_actions.extend(actions)

        return np.array(elite_obs), np.array(elite_actions)

    def learn(self, observations, actions):
        """
        Train the policy network using supervised learning on elite episode data.

        This method performs one training step by computing the cross-entropy loss
        between the network's action predictions and the actual actions taken in
        elite episodes. The network parameters are updated using gradient descent.

        Args:
            observations (np.ndarray): Array of state observations from elite episodes.
            actions (np.ndarray): Array of corresponding actions from elite episodes.

        Returns:
            float: The computed loss value after the training step.
        """
        # 1. Convert observations and actions to tensors, they are already batched
        observations_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)

        # 2. Pass observations and get raw logits
        pred_actions = self.policy.forward(observations_tensor)

        # 3. Do a backward pass
        # 3.1. First always zero out the gradients
        self.optimizer.zero_grad()
        # 3.2. Calculate cross-entropy loss (softmax applied internally)
        loss: torch.Tensor = self.loss(pred_actions, actions_tensor)
        # 3.3 Backpropogate the loss
        loss.backward()
        # 3.4. Update the weights
        self.optimizer.step()
        return loss.item()

    def save(self, save_path: str):
        """
        Save the current policy network state to disk.

        This method creates the target directory if it doesn't exist and saves
        the policy network's state dictionary to a file named 'policy.pth'.

        Args:
            save_path (str): Directory path where the policy should be saved.
                The method will create this directory if it doesn't exist.
        """
        print("--- Saving Policy ---")
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.policy.state_dict(), f"{save_path}/policy.pth")
        print(f"Policy saved in {save_path}")

    def load(self, load_path: str):
        """
        Load a previously saved policy network state from disk.

        This method attempts to load the policy network's state dictionary from
        a file named 'policy.pth' in the specified directory. If the file is not
        found, it prints a warning and continues with the current (untrained) policy.

        Args:
            load_path (str): Directory path containing the saved policy file.
                The method expects to find 'policy.pth' in this directory.
        """
        try:
            self.policy.load_state_dict(torch.load(f"{load_path}/policy.pth"))
        except FileNotFoundError:
            print(f"Policy file not found in {load_path}. Using untrained policy.")

    def play_one_episode(self, env: gym.Env = None, deterministic: bool = False, render: bool = False):
        """
        Play a complete episode in the environment using the current policy.

        This method runs a full episode from start to finish, collecting the sequence
        of states and actions taken, along with the total reward accumulated. The
        episode can be played deterministically or stochastically, and optionally
        rendered for visualization.

        Args:
            env (gym.Env, optional): Environment to play the episode in. If None,
                uses the agent's default environment. Defaults to None.
            deterministic (bool, optional): Whether to use deterministic action
                selection. Defaults to False.
            render (bool, optional): Whether to render the environment during play.
                Defaults to False.

        Returns:
            tuple: A tuple containing:
                - history (list): List of (observation, action) tuples for each step.
                - total_reward (float): Total reward accumulated during the episode.
        """
        if env is None:
            env = self.env

        # Get initial observation, note no seed is needed as it was seeded in the base class
        observation, _ = env.reset()
        # A list to store the history of (state, action) tuples
        history = []
        # A flag whether the episode has finished
        done = False
        # Variable to track the total episode's reward
        total_reward = 0

        # Loop while it is still not done
        while not done:
            # Get action from our policy
            action = self.choose_action(observation=observation, deterministic=deterministic)
            # Apply our action and receive the feedback
            next_observation, reward, terminated, truncated, _ = env.step(action)
            # Was the step terminal?
            done = terminated or truncated
            # Store the tuple in the history
            history.append((observation, action))
            # If asked, then render the state
            if render:
                env.render()
            # Transit to the next observation
            observation = next_observation
            total_reward += reward
        return history, total_reward

    def train(self, num_epochs: int, num_episodes: int, percentile: float):
        """
        Train the agent using the Cross-Entropy Method (CEM).

        This method implements the main CEM training loop. For each epoch, it collects
        data from multiple episodes, selects elite episodes based on their performance,
        and trains the policy network on the elite episode data. Training continues
        until either the maximum number of epochs is reached or the solved threshold
        is achieved.

        Args:
            num_epochs (int): Maximum number of training epochs to run.
            num_episodes (int): Number of episodes to collect data from in each epoch.
            percentile (float): Percentile threshold for selecting elite episodes.
                Higher values result in more selective elite episode filtering.

        Note:
            Training will terminate early if the mean reward of an epoch exceeds
            the solved_threshold specified during initialization.
        """

        pbar = tqdm(range(num_epochs), desc="Training", postfix={"mean_reward": 0})

        for e in pbar:
            # Collect data from multiple episodes
            episodes_data = []

            # Sequential processing for classic environments
            # Plan N episodes and collect data
            for _ in range(num_episodes):
                history, total_reward = self.play_one_episode()

                # Append to the external list total reward
                self.train_rewards.append(total_reward)

                # Extract observations and actions from history
                episode_observations = [step[0] for step in history]
                episode_actions = [step[1] for step in history]

                # Store episode data: (states, actions, total_reward)
                episodes_data.append((episode_observations, episode_actions, total_reward))

            episodic_rewards = [episode[2] for episode in episodes_data]

            mean_reward = np.mean(episodic_rewards)
            pbar.set_postfix_str(f"mean_reward: {mean_reward:.2f}")

            # Select elite episodes based on total episode rewards
            elite_observations, elite_actions = self.select_elite_episodes(
                episodes_data=episodes_data, percentile=percentile
            )

            # Only train if we have elite data
            if len(elite_observations) > 0:
                self.learn(elite_observations, elite_actions)

            # Check for early termination
            if mean_reward > self.solved_threshold:
                pbar.close()
                print(f"Terminated after {e+1} steps with mean reward {mean_reward:.2f}")
                break
