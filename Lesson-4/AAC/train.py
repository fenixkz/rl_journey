import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from model import Agent
import torch.optim as opt
SEED = 24 # This number is used to ensure reproducibility, use any integer you like

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
set_seed()  # Set the seed for reproducibility


env = gym.make("CartPole-v1")
device = "cuda" if torch.cuda.is_available() else "cpu"
agent = Agent(env.observation_space.shape[0], env.action_space.n, device).to(device)

actor_optimizer = opt.Adam(agent.actor.parameters(), lr=3e-4)
critic_optimizer = opt.Adam(agent.critic.parameters(), lr=1e-2) # High learning rate for critic, because otherwise it will learn very slowly

n_episode = 2000
gamma = 0.99
critic_loss_weight = 1.0 
ENTROPY_BETA = 0.0      # Coefficient for the entropy bonus

episode_rewards = []
for e in range(n_episode):
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

    td_target = rewards + gamma * v_next
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

    if e % 100 == 0:
        print(f"Episode {e}, Total reward: {episode_rewards[-1]:.3f}")

import matplotlib.pyplot as plt
# Calculate moving average with window size of 50
window_size = 50
if len(episode_rewards) >= window_size:
    moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    # Create x-axis for moving average (offset by window_size-1)
    moving_avg_x = np.arange(window_size-1, len(episode_rewards))
else:
    moving_avg = episode_rewards
    moving_avg_x = np.arange(len(episode_rewards))

plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, alpha=0.3, color='lightblue', label='Episode Rewards')
plt.plot(moving_avg_x, moving_avg, color='darkblue', linewidth=2, label=f'Moving Average (window={window_size})')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.title(f"REINFORCE Performance on CartPole-v1\n AAC")
plt.savefig(f"reinforce_cartpole_AAC.png")
plt.show()