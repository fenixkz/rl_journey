import json
import os
import random
from abc import ABC, abstractmethod
from typing import Dict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich import print as pprint


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


class BaseAgent(ABC):
    def __init__(self, env: gym.Env, solved_threshold: float, seed: int = 224):
        self.env = env
        self.solved_threshold = solved_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        set_seed(seed)
        # First reset for reproducing same results
        env.reset(seed=seed)
        env.action_space.seed(seed)
        # A list to store episodic rewards during training
        self.train_rewards = []
        # A list to store episodic rewards during evaluation
        self.val_rewards = []
        # A window to calculate the mean of rewards
        self.window_size = 100

    def __del__(self):
        self.env.close()

    def get_figure(self):
        """
        Generates a matplotlib figure plotting training rewards and optional evaluation rewards.

        Returns:
            matplotlib.figure.Figure: The generated plot figure.
        """
        fig, ax = plt.subplots(figsize=(12, 6))  # Create a figure and axes

        # --- Plotting Training Rewards ---
        # Plot transparent raw rewards for every episode
        ax.plot(self.train_rewards, alpha=0.2, color="lightblue", label="Training Episode Rewards")

        # Calculate and plot the moving average for training rewards
        if len(self.train_rewards) >= self.window_size:
            moving_avg = np.convolve(self.train_rewards, np.ones(self.window_size) / self.window_size, mode="valid")
            moving_avg_x = np.arange(self.window_size - 1, len(self.train_rewards))
            ax.plot(
                moving_avg_x,
                moving_avg,
                color="darkblue",
                linewidth=2,
                label=f"Training Moving Average (window={self.window_size})",
            )

        # --- Plotting Optional Evaluation Rewards ---
        if len(self.val_rewards) > 0:

            # 1. Calculate the x-coordinates for each evaluation point
            eval_x_coords = np.linspace(0, len(self.train_rewards) - 1, len(self.val_rewards))

            # 2. Plot the interpolated line and highlight the actual points
            ax.plot(
                eval_x_coords,
                self.val_rewards,
                "-",
                color="limegreen",
                linewidth=2,
                label="Evaluation Rewards (Interpolated)",
            )
            ax.plot(eval_x_coords, self.val_rewards, "o", color="limegreen", markersize=8, markeredgecolor="black")

        # --- General Plot Formatting ---
        ax.axhline(
            y=self.solved_threshold, color="red", linestyle="--", label=f"Solved Threshold ({self.solved_threshold})"
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Episodic Rewards")
        ax.set_title("Training and Evaluation Performance")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig

    def save_as_json(self, save_dir: str, config: Dict):
        """
        Saves a config into a json format file
        """
        save_path = os.path.join(save_dir, "config.json")
        with open(save_path, "w") as json_file:
            json.dump(config, json_file, indent=4)

    def save_progress(self, save_path: str, config: Dict):
        """A non-interactive function to save model and plot to file."""

        if not self.train_rewards:
            pprint("[bold red] History of rewards is empty, nothing to save.")
            return

        pprint("\nSaving model and plotting results to file...")

        # 1. Save the agent's policy
        self.save(save_path)

        # 2. Save config
        self.save_as_json(save_dir=save_path, config=config)

        # 3. Save the plot of rewards
        fig = self.get_figure()
        pprint("Generated the figure")
        save_file_path = os.path.join(save_path, "rewards.jpg")
        try:
            fig.savefig(save_file_path)
            pprint(f"Plot saved to {save_file_path}")
        except Exception as e:
            pprint(f"Could not save plot: {e}")
        finally:
            plt.close(fig)

    @abstractmethod
    def choose_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Given an observation (state), use internal policy to choose an action

        Args:
            obs: Observation from the environment
            deterministic: Whether or not the best action is given or sampled, defaults to False

        Returns:
            Action (int)
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, **kwargs):
        """
        Main method to train the agent

        Args:
            train_rewards: An empty list which will by populated by each episode's rewards during training
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, save_path: str):
        """
        Method to save the main module behind decision-making (neural network, lookup table, policy, dqn, etc.)

        Args:
            save_path: A path to the folder where the module will be saved
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, load_path: str):
        """
        Method to load pre-trained module behind decision-making

        Args:
            load_path: A path to the folder where the module is saved
        """
        raise NotImplementedError
