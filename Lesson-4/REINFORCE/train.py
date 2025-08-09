import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from agent import Agent
import torch.optim as opt
import matplotlib.pyplot as plt
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.classic_config import CLASSIC_ENV_CONFIG
import random
from tqdm import tqdm

SEED = 24  # This number is used to ensure reproducibility, use any integer you like




def create_agent(obs_space, action_space, device) -> torch.nn.Module:
    """
    Create an agent based on the configuration
    
    Args:
        obs_space: Observation space size
        action_space: Action space size
        device: Device to place the agent on
    
    Returns:
        agent: The created agent
    """
    
    agent = Agent(obs_space, action_space, hidden_size=64, dropout_rate=0.1)
    
    return agent.to(device)

def train_environment(env_name, config, verbose=True):
    """
    Train REINFORCE on a specific environment
    
    Args:
        env_name: Name of the gymnasium environment
        config: Configuration dictionary for the environment
        normalize_returns: Whether to normalize returns
        use_entropy: Whether to use entropy regularization
        verbose: Whether to print training progress
    
    Returns:
        episode_rewards: List of episode rewards
        agent: Trained agent
    """
    set_seed()  # Set the seed for reproducibility
    
    try:
        env = gym.make(env_name)
    except Exception as e:
        print(f"Error creating environment {env_name}: {e}")
        print("Make sure the environment is installed. For LunarLander, you may need: pip install gymnasium[box2d]")
        return None, None
    
    if verbose:
        print(f"Environment: {env_name}")
        print(f"Description: {config['description']}")
        print(f"Observation space: {env.observation_space.shape}, Action space: {env.action_space.n}")
        print(f"Training for {config['max_episodes']} episodes, target: {config['solved_threshold']}")
        print("-" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create agent with environment-specific configuration
    agent = create_agent(env.observation_space.shape[0], env.action_space.n, device)
    optimizer = opt.Adam(agent.parameters(), lr=3e-4)
    env.action_space.seed(SEED)  # Set action space seed

    episode_rewards = []
    best_avg_reward = float('-inf')
    solved_episode = None
    
    normalize_returns = config['use_normalization']
    use_entropy = config['use_entropy']

    # Create progress bar
    pbar = tqdm(range(config['max_episodes']), desc=f"Training {env_name}", disable=not verbose)
    # First reset for reproducing same results
    env.reset(seed = SEED)
    env.action_space.seed(SEED)
    for e in pbar:
        state, _ = env.reset(seed=SEED+e)  # Reset with different seed for each episode
        done = False
        
        # Data structures to store log probabilities, entropies, and rewards
        log_probs = []
        entropies = []
        rewards = []

        while not done:
            # State is internally converted to a tensor by the agent
            logits = agent(state) # Logits are the raw scores for each action, shape: (batch_size, action_space)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1) # Convert logits to probabilities via softmax operator, shape: (batch_size, action_space)
            
            # Create a categorical distribution from the probabilities
            # This distribution will be used to sample actions and compute log probabilities
            action_dist = torch.distributions.Categorical(probs) # Categorical distribution for sampling actions, shape: (batch_size, action_space)
            
            # Sample an action from the distribution
            # action_dist.sample() returns a tensor of shape (batch_size,) with the sampled actions
            # We use item() to get the scalar value from the tensor
            action = action_dist.sample() # Sample an action from the distribution, shape: (batch_size,)
            # Apply the action to the environment
            next_state, reward, terminated, truncated, _ = env.step(action.item()) # Use .item() to extract the integer from the tensor
            done = terminated or truncated
            
            if use_entropy:
                # Use entropy to encourage exploration
                # Entropy is calculated as sum of -p * log(p) for each action probability, but we can use the built-in method
                entropies.append(action_dist.entropy()) # action_dist.entropy() has shape (batch_size,)
            
            # Store log probabilities and rewards
            log_probs.append(action_dist.log_prob(action)) # We can also use built-in method to compute log probabilities 
            rewards.append(reward) # Store the reward for this step
            state = next_state # Transition to the next state

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            # Use backward pass to compute returns for each step
            G = r + 0.99 * G
            returns.insert(0, G)
        
        # Convert to tensors
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        log_probs = torch.stack(log_probs).to(device)
        
        # Normalize returns if specified
        # Normalization helps to stabilize the training by reducing the huge variation in returns
        if normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute the loss 
        # The loss is the negative log probability of the actions taken, weighted by the returns
        # This is the REINFORCE algorithm
        # We want to maximize the expected return, so we minimize the negative log probability
        loss = -(log_probs * returns).sum()
        
        if use_entropy:
            entropies = torch.stack(entropies).to(device)
            loss -= 0.01 * entropies.sum() # sum or mean? sum is more common in literature, but mean is also used

        episode_rewards.append(sum(rewards))
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check if environment is solved
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
            
            if avg_reward >= config['solved_threshold'] and solved_episode is None:
                solved_episode = e
                if verbose:
                    pbar.write(f"🎉 Environment solved at episode {e}! Average reward: {avg_reward:.2f}")
                    break  # Stop training if solved
        # Print progress
        if verbose and e % 200 == 0:
            recent_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            pbar.set_postfix({
                                'Loss': f"{loss.item():.3f}",
                                'Reward': f"{sum(rewards):.1f}",
                                'Avg(100)': f"{recent_avg:.1f}",
                                'Best': f"{best_avg_reward:.1f}"
                            })

    env.close()
    pbar.close()
    return episode_rewards, agent


def plot_single_environment(env_name, rewards, config, save_dir="figures"):
    """Plot results for a single environment"""
    os.makedirs(save_dir, exist_ok=True)
    
    window_size = 50
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        moving_avg_x = np.arange(window_size-1, len(rewards))
    else:
        moving_avg = rewards
        moving_avg_x = np.arange(len(rewards))

    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, color='lightblue', label='Episode Rewards')
    plt.plot(moving_avg_x, moving_avg, color='darkblue', linewidth=2, 
             label=f'Moving Average (window={window_size})')
    
    # Add solved threshold line
    plt.axhline(y=config['solved_threshold'], color='red', linestyle='--', 
                label=f'Solved Threshold ({config["solved_threshold"]})')
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    # Enhanced title with agent info
    agent_info = f"{config.get('agent_type', 'basic')} ({config.get('num_layers', 2)} layers)"
    plt.title(f"REINFORCE Performance on {env_name}\n{config['description']} | Agent: {agent_info}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    config_str = f"Norm={config['use_normalization']}_Ent={config['use_entropy']}"
    filename = f"reinforce_{env_name.replace('-', '_').lower()}_{config_str}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

def plot_multi_environment_results(results, save_dir="figures"):
    """Plot results for all environments in subplots"""
    os.makedirs(save_dir, exist_ok=True)
    
    n_envs = len([r for r in results.values() if r['rewards'] is not None])
    if n_envs == 0:
        print("No successful training results to plot.")
        return
    
    cols = 2
    rows = (n_envs + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    for env_name, data in results.items():
        if data['rewards'] is None:  # Skip failed environments
            continue
            
        rewards = data['rewards']
        config = data['config']
        window_size = 50
        
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            moving_avg_x = np.arange(window_size-1, len(rewards))
        else:
            moving_avg = rewards
            moving_avg_x = np.arange(len(rewards))
        
        ax = axes[plot_idx] if n_envs > 1 else axes
        ax.plot(rewards, alpha=0.3, color='lightblue', label='Episode Rewards')
        ax.plot(moving_avg_x, moving_avg, color='darkblue', linewidth=2, 
                label=f'Moving Average (window={window_size})')
        
        # Add solved threshold line
        ax.axhline(y=config['solved_threshold'], color='red', linestyle='--', 
                   label=f'Solved ({config["solved_threshold"]})')
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        
        # Add agent type to title
        agent_info = f"{config.get('agent_type', 'basic')}"
        ax.set_title(f"REINFORCE on {env_name}\n({agent_info})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    config_str = f"Norm={config['use_normalization']}_Ent={config['use_entropy']}_Enhanced"
    plt.savefig(os.path.join(save_dir, f"reinforce_multi_environment_{config_str}.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()

def train_all_environments():
    """
    Train REINFORCE on all configured environments
    
    Args:
        selected_envs: List of environment names to train on. If None, train on all.
    
    Returns:
        results: Dictionary containing training results for each environment
    """
    
    results = {}
    
    print("🚀 Starting Multi-Environment REINFORCE Training (Enhanced)")

    print("=" * 80)
    
    for env_name in ENVIRONMENTS.keys():
            
        config = ENVIRONMENTS[env_name]
        print(f"\n🎯 Training on {env_name}")
        print("=" * 60)
        
        episode_rewards, trained_agent = train_environment(
            env_name, config, 
        )
        
        if episode_rewards is not None:
            # Evaluate performance
            solved, final_avg, solve_episode = evaluate_performance(episode_rewards, config)
            
            results[env_name] = {
                'rewards': episode_rewards,
                'agent': trained_agent,
                'config': config,
                'solved': solved,
                'final_avg_reward': final_avg,
                'solve_episode': solve_episode
            }
            
            # Plot individual environment results
            plot_single_environment(env_name, episode_rewards, config)
        else:
            results[env_name] = {
                'rewards': None,
                'agent': None,
                'config': config,
                'solved': False,
                'final_avg_reward': None,
                'solve_episode': None
            }
    
    # Plot comparative results
    if len([r for r in results.values() if r['rewards'] is not None]) > 1:
        plot_multi_environment_results(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("🏆 TRAINING SUMMARY (Enhanced)")
    print("=" * 80)
    
    for env_name, data in results.items():
        if data['rewards'] is None:
            print(f"{env_name:20s}: ❌ Failed to create environment")
            continue
            
        status = "✅ SOLVED" if data['solved'] else "❌ Not Solved"
        solve_info = f"(Episode {data['solve_episode']})" if data['solve_episode'] else ""
        agent_type = data['config'].get('agent_type', 'basic')
        print(f"{env_name:20s}: {status:12s} Final Avg: {data['final_avg_reward']:8.2f} {solve_info} | Agent: {agent_type}")
    
    return results

if __name__ == "__main__":
    # Train on all environments
    results = train_all_environments()
    
    print("\n🎉 Multi-environment training completed!")
    print("Check the 'figures' directory for visualization plots.")