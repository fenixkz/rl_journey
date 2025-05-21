import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from model import Agent
import torch.optim as opt


env = gym.make("CartPole-v1")
device = "cuda" if torch.cuda.is_available() else "cpu"
agent = Agent(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer= opt.Adam(agent.parameters(), lr=0.0001)

n_episode = 10000
gamma = 0.99

for e in range(n_episode):
    state, info = env.reset()
    done = False
    log_probs = []
    rewards = []
    while not done:
        state = torch.FloatTensor(state).to(device)
        logits = agent(state)
        probs = F.softmax(logits)
        
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        log_probs.append(action_dist.log_prob(action))
        rewards.append(reward)
        state = next_state
    
    # Compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    log_probs = torch.stack(log_probs).to(device)
    # Normalize for better stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = -(log_probs * returns).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 100 == 0:
        print(f"Episode {e}, Loss: {loss.item():.3f} Total reward: {sum(rewards):.3f}")