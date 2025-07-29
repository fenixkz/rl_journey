import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from agent import Agent
import torch.optim as opt
import matplotlib.pyplot as plt
import os

SEED = 24  # This number is used to ensure reproducibility, use any integer you like

# Environment configurations with specific hyperparameters
ENVIRONMENTS = {
    'CartPole-v1': {
        'max_episodes': 2000,
        'solved_threshold': 195.0,  # Average reward over 100 episodes
        'lr': 1e-3,
        'gamma': 0.99,
        'hidden_size': 64,
        'num_layers': 1,  # Simple environment, single layer is enough
        'agent_type': 'basic',
        'description': 'Classic control - balance pole on cart'
    },
    'MountainCar-v0': {
        'max_episodes': 3000,  # Harder environment, needs more episodes
        'solved_threshold': -110.0,  # MountainCar has negative rewards
        'lr': 1e-3,
        'gamma': 0.99,
        'hidden_size': 128,
        'num_layers': 2,  # More complex dynamics
        'agent_type': 'basic',
        'description': 'Get car to top of mountain using momentum'
    },
    'LunarLander-v3': {
        'max_episodes': 2000,
        'solved_threshold': 200.0,
        'lr': 5e-4,  # Lower learning rate for stability
        'gamma': 0.99,
        'hidden_size': 128,
        'num_layers': 2,
        'agent_type': 'dropout',  # Use dropout for regularization
        'description': 'Land spacecraft safely on moon surface'
    },
    'Acrobot-v1': {
        'max_episodes': 3000,
        'solved_threshold': -100.0,
        'lr': 1e-3,
        'gamma': 0.99,
        'hidden_size': 128,
        'num_layers': 2,
        'agent_type': 'basic',
        'description': 'Swing up underactuated pendulum'
    }
}

# Global hyperparameters
NORMALIZE_RETURNS = False  # Set to False, True
USE_ENTROPY = True         # Set to False, True
ENTROPY_BETA = 0.01        # Coefficient for the entropy bonus

def set_seed():
    """
    Set the seed for the environment and PyTorch.
    """
    # Set NumPy random seed
    np.random.seed(SEED)

    # Set PyTorch random seed
    torch.manual_seed(SEED)

    # If using CUDA, also set CUDA random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups
        # Make CUDA operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_agent(obs_space, action_space, config, device):
    """
    Create an agent based on the configuration
    
    Args:
        obs_space: Observation space size
        action_space: Action space size
        config: Environment configuration
        device: Device to place the agent on
    
    Returns:
        agent: The created agent
    """
    agent_type = config.get('agent_type', 'basic')
    hidden_size = config['hidden_size']
    num_layers = config.get('num_layers', 2)
    
    if agent_type == 'basic':
        agent = Agent(obs_space, action_space, hidden_size, num_layers)
    elif agent_type == 'dropout':
        agent = Agent(obs_space, action_space, hidden_size, num_layers, dropout_rate=0.2)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent.to(device)

def train_environment(env_name, config, normalize_returns=False, use_entropy=True, verbose=True):
    """
    Train AAC on a specific environment
    
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
        print(f"Hyperparameters: lr={config['lr']}, gamma={config['gamma']}, hidden_size={config['hidden_size']}")
        print(f"Agent type: {config.get('agent_type', 'basic')}, Layers: {config.get('num_layers', 2)}")
        print(f"Training for {config['max_episodes']} episodes, target: {config['solved_threshold']}")
        print("-" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create agent with environment-specific configuration
    agent = create_agent(env.observation_space.shape[0], env.action_space.n, config, device)
    actor_optimizer = opt.Adam(agent.actor.parameters(), lr=3e-4)
    critic_optimizer = opt.Adam(agent.critic.parameters(), lr=1e-2)
    env.action_space.seed(SEED)  # Set action space seed

    episode_rewards = []
    best_avg_reward = float('-inf')
    solved_episode = None
    
    for e in range(config['max_episodes']):
        state, info = env.reset(seed=SEED+e)
        done = False
        total_reward = 0

        # Data structures to store log probabilities, entropies, and rewards
        log_probs = []
        entropies = []
        states = []
        next_states = []
        rewards = []  # Store rewards for the episode
        dones = []  # Store done flags for the episode
        while not done:
            # Note: states are converted to tensors and moved to the correct device internally in agent's methods
            # Get probability distribution over actions
            probs = agent.act(state) # probs shape: [1, action_space]

            ## Sample an action
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample() # action shape: [1], single action sampled from the distribution

            # Make a step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action.item()) # action.item() to get the scalar value from the tensor
            done = terminated or truncated
            total_reward += reward

            # Compute log probability of the action
            log_prob = action_dist.log_prob(action) # log_prob shape: [1] (tensor)
            # Compute the entropy of the action distribution
            entropy = action_dist.entropy() # entropy shape: [1] (tensor)

            states.append(state)  # Store the current state
            next_states.append(next_state)  # Store the next state
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            entropies.append(entropy)

            state = next_state
        # End of episode, store the total reward
        episode_rewards.append(sum(rewards))
        
        # Convert lists to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).view(-1, 1)  # Convert rewards to tensor, shape: [n_steps, 1]
        dones = torch.tensor(dones, dtype=torch.float32, device=device).view(-1, 1)   # Convert dones to tensor, shape: [n_steps, 1]
        log_probs = torch.stack(log_probs).to(device)  # Stack log probabilities into a tensor, shape: [n_steps, 1]
        entropies = torch.stack(entropies).to(device)  # Stack entropies into a tensor, shape: [n_steps, 1]
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)  # Convert states to tensor, shape: [n_steps, state_dim]
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)  # Convert next states to tensor, shape: [n_steps, state_dim]
        
        # Compute state values using the Critic network
        v_s = agent.evaluate(states)  # Evaluate current state values, shape: [n_steps, 1]
        
        # Compute TD-target using the 1 step return
        with torch.no_grad():
            v_next = agent.evaluate(next_states)
            # For terminal states, next value should be 0
            v_next = v_next * (1 - dones)

        td_target = rewards + config['gamma'] * v_next
        advantage = td_target - v_s  # advantage shape: [N, 1] (tensor)
        
        # Compute losses
        # Actor loss: -log_prob * Advantage.
        # Detach advantage for actor loss
        actor_loss = -(log_probs * advantage.detach()).mean() - ENTROPY_BETA*entropies.mean()
        critic_loss = F.mse_loss(v_s, td_target.detach())  # MSE loss between predicted and target values, shape: scalar
        
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=0.5)
        actor_optimizer.step()
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=0.5)
        critic_optimizer.step()
        total_loss = actor_loss.detach().item() + critic_loss.detach().item()  # Total loss is the sum of actor and critic losses
        # Check if environment is solved
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
            
            if avg_reward >= config['solved_threshold'] and solved_episode is None:
                solved_episode = e
                if verbose:
                    print(f"🎉 Environment solved at episode {e}! Average reward: {avg_reward:.2f}")
                    break  # Stop training if solved
        # Print progress
        if verbose and e % 200 == 0:
            recent_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Episode {e:4d}, Loss: {total_loss:8.3f}, Total reward: {rewards.sum().item():8.3f}, Avg(100): {recent_avg:8.3f}")

    env.close()
    
    return episode_rewards, agent

def evaluate_performance(env_name, rewards, config):
    """
    Evaluate if the agent has solved the environment
    
    Args:
        env_name: Name of the environment
        rewards: List of episode rewards
        config: Environment configuration
    
    Returns:
        solved: Boolean indicating if environment is solved
        avg_reward: Average reward over last 100 episodes
        solve_episode: Episode where environment was first solved (or None)
    """
    if len(rewards) < 100:
        return False, np.mean(rewards), None
    
    # Check when environment was first solved
    solve_episode = None
    for i in range(100, len(rewards)):
        avg_reward = np.mean(rewards[i-100:i])
        if avg_reward >= config['solved_threshold']:
            solve_episode = i
            break
    
    final_avg = np.mean(rewards[-100:])
    solved = final_avg >= config['solved_threshold']
    
    return solved, final_avg, solve_episode

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
    plt.title(f"AAC Performance on {env_name}\n{config['description']} | Agent: {agent_info}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    config_str = f"Norm={NORMALIZE_RETURNS}_Ent={USE_ENTROPY}_Enhanced"
    filename = f"AAC_{env_name.replace('-', '_').lower()}_{config_str}.png"
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
        ax.set_title(f"AAC on {env_name}\n({agent_info})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    config_str = f"Norm={NORMALIZE_RETURNS}_Ent={USE_ENTROPY}_Enhanced"
    plt.savefig(os.path.join(save_dir, f"AAC_multi_environment_{config_str}.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()

def train_all_environments(selected_envs=None):
    """
    Train AAC on all configured environments
    
    Args:
        selected_envs: List of environment names to train on. If None, train on all.
    
    Returns:
        results: Dictionary containing training results for each environment
    """
    if selected_envs is None:
        selected_envs = list(ENVIRONMENTS.keys())
    
    results = {}
    
    print("🚀 Starting Multi-Environment AAC Training (Enhanced)")
    print(f"Configuration: Normalize Returns = {NORMALIZE_RETURNS}, Use Entropy = {USE_ENTROPY}")
    print(f"Entropy Beta = {ENTROPY_BETA}, Seed = {SEED}")
    print("=" * 80)
    
    for env_name in selected_envs:
        if env_name not in ENVIRONMENTS:
            print(f"Warning: Environment {env_name} not in configuration. Skipping.")
            continue
            
        config = ENVIRONMENTS[env_name]
        print(f"\n🎯 Training on {env_name}")
        print("=" * 60)
        
        episode_rewards, trained_agent = train_environment(
            env_name, config, 
            normalize_returns=NORMALIZE_RETURNS,
            use_entropy=USE_ENTROPY
        )
        
        if episode_rewards is not None:
            # Evaluate performance
            solved, final_avg, solve_episode = evaluate_performance(env_name, episode_rewards, config)
            
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
    
    # Optional: Train on specific environments only
    # results = train_all_environments(['CartPole-v1', 'MountainCar-v0'])
    
    print("\n🎉 Multi-environment training completed!")
    print("Check the 'figures' directory for visualization plots.")