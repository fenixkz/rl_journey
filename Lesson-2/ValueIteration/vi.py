import gymnasium as gym
import numpy as np

# Define the env
env = gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=True)

# Get number of states (8x8=64)
n_states = env.observation_space.n
# Get number of possible actions (4)
n_actions = env.action_space.n
# Unwrap the env to get transition probabilities
base_env = env.unwrapped
# Get a matrix of transition probabilities
transition_prob = base_env.P
# Initialize a dictionary that hold a state-value for each state
# V*(s) for all s in S, initialized as 0 in the beginning
state_values = {s: 0 for s in range(n_states)}
# Discount factor
gamma = 0.9


def get_Q(state, action):
    """
    A function to compute the Q*(s,a)
    """
    Q = 0
    # Per formula in theory.md
    for prob, next_state, reward, _ in transition_prob[state][action]:
        Q += prob * (reward + gamma * state_values[next_state])
    return Q


def compute_V(state):
    """
    A function to compute V*(s)
    """
    # Per formula in theory.md
    return max(get_Q(state, action) for action in range(n_actions))


def choose_action(state):
    """
    A function to choose the action that maximizes the Q*(s,a)
    """
    # Chooses the best action that corresponds to the highest Q value
    return max(range(n_actions), key=lambda action: get_Q(state, action))


# Total number of iterations to perform
num_iter = 100
for i in range(num_iter):
    # Re-calculate state-values
    new_state_values = {s: compute_V(s) for s in range(n_states)}
    # Get a maximum difference between old estimates and new
    diff = max(abs(new_state_values[s] - state_values[s]) for s in range(n_states))
    print(f"Iteration {i}, diff: {diff}")
    state_values = new_state_values
    # If the difference is low, it means we are close to convergence
    if diff < 0.001:
        break

n_episodes = 100
n_steps = 200
# Play the episode with our estimates of values
rewards = []
for i in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0
    for j in range(n_steps):
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        total_reward += reward
        if terminated or truncated:
            break
    rewards.append(total_reward)

print(f"Average reward per {n_episodes} is {np.mean(rewards)}")
