import matplotlib.pyplot as plt
from typing import Iterable
import numpy as np

def get_figure(mean_rewards: Iterable, std_rewards: Iterable, num_episodes: int):
    mean_rewards = np.array(mean_rewards)
    std_rewards = np.array(std_rewards)
    n_episodes = len(mean_rewards)
    fig, ax = plt.subplots(figsize=(12, 6)) # Create a figure and axes

    # Plot the mean reward line
    ax.plot(range(n_episodes), mean_rewards, color='green', label='Mean Training Episodic Reward')

    # Plot the standard deviation area
    # y1 is the lower bound, y2 is the upper bound
    ax.fill_between(range(n_episodes), 
                    mean_rewards - std_rewards, 
                    mean_rewards + std_rewards, 
                    color='green', 
                    alpha=0.2,  # Use alpha for transparency
                    label='Standard Deviation')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean Rewards')
    ax.set_title(f"Mean of rewards of {num_episodes} over total number of training episodes")
    ax.grid(True)
    return fig
    
