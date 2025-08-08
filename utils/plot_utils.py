import matplotlib.pyplot as plt
from typing import Iterable
import numpy as np

def get_figure(all_rewards: Iterable, solved_threshold: float, window_size: int = 50):

    if len(all_rewards) >= window_size:
        moving_avg = np.convolve(all_rewards, np.ones(window_size)/window_size, mode='valid')
        moving_avg_x = np.arange(window_size-1, len(all_rewards))
    else:
        moving_avg = all_rewards
        moving_avg_x = np.arange(len(all_rewards))


    fig, ax = plt.subplots(figsize=(12, 6)) # Create a figure and axes

    # Plot the mean reward line
    ax.plot(moving_avg_x, moving_avg, color='darkblue', linewidth=2, 
             label=f'Moving Average (window={window_size})')
    # Plot transparent all real rewards
    ax.plot(all_rewards, alpha=0.3, color='lightblue', label='Episode Rewards')
    
    ax.axhline(y=solved_threshold, color='red', linestyle='--', 
                label=f'Solved Threshold ({solved_threshold})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Episodic Rewards')
    ax.set_title(f"All rewards with mean line of {window_size} episodes")
    ax.grid(True, alpha=0.3)
    return fig
    
