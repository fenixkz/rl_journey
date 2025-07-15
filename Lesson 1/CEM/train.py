import torch
import gymnasium as gym
import numpy as np
import time
import sys

from agent import Agent
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.plot_utils import get_figure

# --- Environment Configurations ---
ENV_CONFIGS = {
    "CartPole-v1": {
        "training_epochs": 100,
        "num_episodes": 50, 
        "threshold_reward": 400,
    },
    "LunarLander-v3": {
        "training_epochs": 300,
        "num_episodes": 50,
        "threshold_reward": 200,
    },
}

def play_one_episode(env: gym.Env, agent: Agent, render: bool = False, deterministic: bool = False):
    state, _ = env.reset()
    history = []
    done = False
    total_reward = 0
    while not done:
        action = agent.choose_action(state=state, deterministic=deterministic)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if render: env.render()
        history.append((state, action))
        state = next_state
        total_reward += reward
    return history, total_reward

def select_elite_episodes(episodes_data, percentile):
    """
    Select elite episodes based on total episode rewards.
    episodes_data: list of (states, actions, total_reward) tuples
    """
    # Sort episodes by total reward
    episodes_data.sort(key=lambda x: x[2], reverse=True)
    
    # Calculate how many episodes to keep
    n_elite = int(len(episodes_data) * (100 - percentile) / 100)
    n_elite = max(1, n_elite)  # Keep at least 1 episode
    
    # Get elite episodes
    elite_episodes = episodes_data[:n_elite]
    
    # Extract all states and actions from elite episodes
    elite_states = []
    elite_actions = []
    
    for states, actions, _ in elite_episodes:
        elite_states.extend(states)
        elite_actions.extend(actions)
    
    return np.array(elite_states), np.array(elite_actions)

if __name__ == "__main__":
    # --- Configuration ---
    env_id = "LunarLander-v3" # "CartPole-v1" "LunarLander-v3" 

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = Agent(obs_dim, act_dim, device).to(device)

    # --- Visualization: Untrained Policy ---

    human_env = gym.make(env_id, render_mode="human") # Create a similar env, but that can visualize the environment using display

    _, total_reward = play_one_episode(human_env, agent, render=True)
    print(f"Total reward achieved by untrained policy: {total_reward}")
    human_env.close()
    # --- End of visualization


    # Use config for the selected environment
    env_config = ENV_CONFIGS.get(env_id, {})
    training_epochs = env_config.get("training_epochs", 100) # Total number of learning steps
    num_episodes = env_config.get("num_episodes", 50) # Number of episodes to play to apply CEM filtering
    threshold_reward = env_config.get("threshold_reward", 200) # A reward after which we consider the env solved

    mean_rewards = [] # To track the evolution of total rewards
    std_rewards = [] # To track the evolution of total rewards
    episodes_in_a_row = 0 # Counter for consecutive good episodes
    terminate_after = 2 # Stop training after this many consecutive good episodes

    pbar = tqdm(range(training_epochs), desc="Training", postfix={"mean_reward": 0})

    for e in pbar:
        # Collect data from multiple episodes
        episodes_data = []
        
        # Play N episodes and gather data
        for episode in range(num_episodes):
            history, total_reward = play_one_episode(env, agent)
            
            # Extract states and actions from history
            episode_states = [step[0] for step in history]
            episode_actions = [step[1] for step in history]
            
            # Store episode data: (states, actions, total_reward)
            episodes_data.append((episode_states, episode_actions, total_reward))
        
        # Calculate mean reward for this training step
        all_rewards = [episode[2] for episode in episodes_data]
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        mean_rewards.append(mean_reward)
        std_rewards.append(std_reward)
        pbar.set_postfix_str(f"mean_reward: {mean_reward:.2f}")
        
        # Select elite episodes based on total episode rewards (top 30%)
        elite_states, elite_actions = select_elite_episodes(episodes_data, percentile=70)
        
        if len(elite_states) > 0:  # Only train if we have elite data
            agent.learn(elite_states, elite_actions)
        
        # Check for early termination
        if mean_reward > threshold_reward:
            episodes_in_a_row += 1
        else:
            episodes_in_a_row = 0
            
        if episodes_in_a_row >= terminate_after:
            pbar.close()
            print(f"Terminated after {e+1} steps with mean reward {mean_reward:.2f}")
            break
        
    # Evaluate policy: use deterministic actions, play 10 episodes and compute average reward
    print("\nEvaluating trained policy...")
    
    # --- Visualization: Trained Policy ---

    human_env = gym.make(env_id, render_mode="human") # Create a similar env, but that can visualize the environment using display

    _, total_reward = play_one_episode(human_env, agent, render=True)
    print(f"Total reward achieved by trained policy: {total_reward}")
    human_env.close()
    # --- End of visualization

    save_path = f"results/{env_id}"
    os.makedirs(save_path, exist_ok=True)

    fig = get_figure(mean_rewards, std_rewards, num_episodes=num_episodes)
    fig.savefig(f"{save_path}/rewards.png")
    plt.show()
    agent.save_policy(env_id)
    env.close()