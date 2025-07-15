import numpy as np
import gymnasium as gym
from qlearning import QLearningAgent
from sarsa import SARSA
from ev_sarsa import EVSARSA
import matplotlib.pyplot as plt
from tdlearning import TDLearning
from collections import deque
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.plot_utils import get_figure
from windy_env import WindyBridgeEnv
GAMMA = 0.9
EPS = 1
EPS_DECAY = 0.999
ALPHA = 0.1
NUM_EPISODES = 5000
MEAN_EPISODES = 50  # Calculate mean reward for that many episodes
PRINT_EVERY = 100 # Print results every

def train(agent: TDLearning, env: gym.Env, is_sarsa: bool = False):
    rewards = deque(maxlen=MEAN_EPISODES)
    mean_rewards = []
    std_rewards = []
    for e in range(NUM_EPISODES):
        state, _  = env.reset()
        if isinstance(state, np.ndarray): state = tuple(state) # To hash we need to cast np.arrays to tuple
        done = False
        total_reward = 0
        next_action = None
        while not done:
            # It is important that we use the same action we sampled for evaluating next Q-value for SARSA 
            if is_sarsa and next_action is not None:
                action = next_action
            else:
                action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if isinstance(next_state, np.ndarray): next_state = tuple(next_state)
            done = terminated or truncated
            next_action = agent.get_action(next_state) # Only needed for SARSA
            agent.update(state, action, reward, next_state, next_action)
            state = next_state
            total_reward += reward
        # Let's decay the epsilon, such that in later epsidodes it used more of greedy policy
        agent.epsilon = max(0.1, agent.epsilon * EPS_DECAY)  # Decay epsilon
        rewards.append(total_reward)
        mean_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))
        if (e+1) % PRINT_EVERY == 0:
            print(f"Episode: {e}, Mean reward: {np.mean(rewards)}")
    return mean_rewards, std_rewards

def train_taxi():
    env_id = "Taxi-v3"
    env = gym.make(env_id)

    action_space = range(env.action_space.n)
    # Initialize agents
    qlearning_agent = QLearningAgent(alpha = ALPHA, gamma = GAMMA, epsilon = EPS, action_space = action_space)
    sarsa_agent = SARSA(alpha = ALPHA, gamma = GAMMA, epsilon = EPS, action_space = action_space)
    evsarsa_agent = EVSARSA(alpha = ALPHA, gamma = GAMMA, epsilon = EPS, action_space = action_space)

    save_path = f"results/{env_id}"
    os.makedirs(save_path,exist_ok=True)
    print("-------- TRAINING Q-LEARNING -------------")
    mean_rewards, std_rewards = train(qlearning_agent, env)
    fig = get_figure(mean_rewards=mean_rewards, std_rewards=std_rewards, num_episodes=MEAN_EPISODES)
    
    fig.savefig(f"{save_path}/q-learning-rewards.jpg")
    plt.close(fig)

    # SARSA
    print("-------- TRAINING SARSA -------------")
    mean_rewards, std_rewards = train(sarsa_agent, env, is_sarsa=True)
    fig = get_figure(mean_rewards=mean_rewards, std_rewards=std_rewards, num_episodes=MEAN_EPISODES)
    
    fig.savefig(f"{save_path}/sarsa-rewards.jpg")
    plt.close(fig)

    # EV-SARSA
    print("-------- TRAINING EV-SARSA -------------")
    mean_rewards, std_rewards = train(evsarsa_agent, env)
    fig = get_figure(mean_rewards=mean_rewards, std_rewards=std_rewards, num_episodes=MEAN_EPISODES)
    
    fig.savefig(f"{save_path}/ev-sarsa-rewards.jpg")
    plt.close(fig)
    env.close()

def visualize_policy(agent, env, title="Agent Policy"):
    """
    Visualizes the learned policy of an agent on a grid environment.

    Args:
        agent: The trained agent with a get_action(state, deterministic=True) method.
        env: The grid world environment instance.
        title (str): The title for the plot.
    """
    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get grid dimensions
    rows, cols = env.grid_size
    
    # Create a grid of policy arrows
    # u for x-direction (columns), v for y-direction (rows)
    u = np.zeros((rows, cols))
    v = np.zeros((rows, cols))

    # Mapping from action index to arrow vector (dx, dy)
    # Note: y-axis is inverted in matplotlib's matrix display, so 'Up' is a negative change in v.
    action_vectors = {
        0: (0, -1),   # Up
        1: (1, 0),   # Right
        2: (0, 1),  # Down
        3: (-1, 0),  # Left
    }

    # Populate the policy grid
    for r in range(rows):
        for c in range(cols):
            state = (r, c)
            # Don't draw arrows in terminal states
            if np.array_equal(state, env.goal_pos) or state in env.chasm_coords:
                continue
            
            # Get the best action from the agent's policy
            action = agent.get_action(state, deterministic=True)
            
            if action in action_vectors:
                dx, dy = action_vectors[action]
                u[r, c] = dx
                v[r, c] = -dy # Invert y for plotting

    # --- Draw the environment background ---
    # Goal (Green)
    ax.add_patch(plt.Rectangle((env.goal_pos[1] - 0.5, env.goal_pos[0] - 0.5), 1, 1, color='green', alpha=0.5))
    # Bridge (Tan)
    for pos in env.bridge_coords:
        ax.add_patch(plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, color='tan'))
    # Chasm (Blue)
    for pos in env.chasm_coords:
        ax.add_patch(plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, color='darkblue', alpha=0.7))

    # --- Draw the policy arrows ---
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    ax.quiver(x, y, u, v, color='black', headwidth=4, headlength=5, scale=25)

    # --- Formatting ---
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis() # Match the (row, col) coordinate system
    ax.set_title(title)
    
    return fig


def train_windy_bridge():
    env_id = "WindyBridge-v0"
    env = WindyBridgeEnv()
    action_space = range(env.action_space.n)
    qlearning_agent = QLearningAgent(alpha = ALPHA, gamma = GAMMA, epsilon = EPS, action_space = action_space)
    sarsa_agent = SARSA(alpha = ALPHA, gamma = GAMMA, epsilon = EPS, action_space = action_space)
    evsarsa_agent = EVSARSA(alpha = ALPHA, gamma = GAMMA, epsilon = EPS, action_space = action_space)

    save_path = f"results/{env_id}"
    os.makedirs(save_path,exist_ok=True)

    print("-------- TRAINING Q-LEARNING -------------")
    mean_rewards, std_rewards = train(qlearning_agent, env)
    fig = visualize_policy(qlearning_agent, env, title="Q-Learning")
    fig.savefig(f"{save_path}/q-learning-policy.jpg")
    plt.close(fig)

    print("-------- TRAINING SARSA -------------")
    mean_rewards, std_rewards = train(sarsa_agent, env, is_sarsa=True)
    fig = visualize_policy(sarsa_agent, env, title="SARSA")
    fig.savefig(f"{save_path}/sarsa-policy.jpg")
    plt.close(fig)

    print("-------- TRAINING EV-SARSA -------------")
    mean_rewards, std_rewards = train(evsarsa_agent, env)
    fig = visualize_policy(evsarsa_agent, env, title="EV-SARSA")
    fig.savefig(f"{save_path}/ev-sarsa-policy.jpg")
    plt.close(fig)

    
if __name__=='__main__':
    train_taxi()
    train_windy_bridge()
