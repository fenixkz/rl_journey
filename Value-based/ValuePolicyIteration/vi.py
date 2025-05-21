import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n

base_env = env.unwrapped

transition_prob = base_env.P

state_values = {s: 0 for s in range(n_states)} # V*(s) for all s in S, initialized as 0 in the beginning
gamma = 0.9 # Discount factor

def get_Q(state, action):
    '''
    A function to compute the Q*(s,a)
    '''
    Q = 0
    for prob, next_state, reward, _ in transition_prob[state][action]:
        Q += prob * (reward + gamma * state_values[next_state])
    return Q

def compute_V(state):
    '''
    A function to compute V*(s)
    '''
    return max(get_Q(state, action) for action in range(n_actions))
    
def choose_action(state):
    '''
    A function to choose the action that maximizes the Q*(s,a)
    '''
    return max(range(n_actions), key=lambda action: get_Q(state, action))


num_iter = 100
for i in range(num_iter):
    new_state_values = {s: compute_V(s) for s in range(n_states)}
    diff = max(abs(new_state_values[s] - state_values[s]) for s in range(n_states))
    print(f"Iteration {i}, diff: {diff}")
    state_values = new_state_values
    if diff < 0.001:
        break

n_episodes = 100
n_steps = 50

rewards = []
for i in range(n_episodes):
    s, _ = env.reset()
    reward_per_episode = 0
    for j in range(n_steps):
        a = choose_action(s)
        nS, r, done, _, _ = env.step(a)
        s = nS
        reward_per_episode += r
        if done:
            break
    rewards.append(reward_per_episode)

print(f"Average reward per {n_episodes} is {np.mean(rewards)}")