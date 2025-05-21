import numpy as np
from qlearning import QLearningAgent
from sarsa import SARSA
from cliff import CliffWorld
import matplotlib.pyplot as plt

def plot_policy(agent, env, title="Policy"):
    """Visualizes the learned policy as arrows on the grid."""
    action_arrows = {
        "left":  (0, -0.4),
        "right": (0, 0.4),
        "up":    (-0.4, 0),
        "down":  (0.4, 0),
    }
    action_names = env.actions

    # Get a grid representation without the agent's current position
    grid = env.get_state_grid()
    fig, ax = plt.subplots(figsize=(env.width * 0.6, env.height * 0.6))
    
    # Use a simple colormap without the get_cmap function
    ax.imshow(grid, cmap='tab10', origin='upper')

    for row in range(env.height):
        for col in range(env.width):
            state = (row, col)
            # Skip plotting arrows in terminal states (cliff, start, goal)
            is_cliff = any(np.array_equal([row, col], pos) for pos in env.cliff_pos)
            is_start = (row == env.height - 1 and col == 0)
            is_goal = (row == env.goal_row and col == env.goal_col)

            if is_cliff or is_start or is_goal:
                continue

            # Get best action for this state based on Q-values
            q_values = [agent.getQ(state, a_name) for a_name in action_names]

            if not q_values or all(q == 0 for q in q_values):
                 # Small black dot for unvisited/zero Q
                 ax.plot(col, row, 'ko', markersize=2)
                 continue

            best_action_idx = np.argmax(q_values)
            best_action_name = action_names[best_action_idx]

            # Get arrow direction
            plot_dy, plot_dx = action_arrows[best_action_name]
            # Plot arrow from center of the cell
            ax.arrow(col, row, plot_dx, plot_dy, head_width=0.2, head_length=0.2, fc='k', ec='k')

    ax.set_title(title)
    # Set ticks and grid
    ax.set_xticks(np.arange(env.width))
    ax.set_yticks(np.arange(env.height))
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
    # Invert Y axis to match numpy array indexing
    plt.gca().invert_yaxis()
    plt.show()


# Environment and Agent Setup
env = CliffWorld()
action_space = env.actions  # ["left", "right", "up", "down"]

gamma = 0.9
epsilon = 0.9  # Lower epsilon for better exploitation
alpha = 0.5
num_episodes = 500

# Q-Learning Training
print("Training Q-Learning Agent...")
agent_q = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, action_space=action_space)
rewards_q = []

for e in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent_q.get_action(state)
        next_state, reward, done = env.step(action)
        agent_q.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    
    agent_q.epsilon = max(0.01, agent_q.epsilon * 0.995)  # Decay epsilon
    rewards_q.append(total_reward)
    
    if (e + 1) % 100 == 0:
        print(f"Q-Learning Episode {e+1}/{num_episodes} - Avg Reward (last 100): {np.mean(rewards_q[-100:]):.2f}")

print(f"Average reward of last 10 episodes for Q-Learning: {np.mean(rewards_q[-10:]):.2f}")

# SARSA Training
print("\nTraining SARSA Agent...")
agent_sarsa = SARSA(alpha=alpha, gamma=gamma, epsilon=epsilon, action_space=action_space)
rewards_sarsa = []

for e in range(num_episodes):
    state = env.reset()
    action = agent_sarsa.get_action(state)  # Get first action
    done = False
    total_reward = 0
    
    while not done:
        next_state, reward, done = env.step(action)
        next_action = agent_sarsa.get_action(next_state)  # Get next action before update
        agent_sarsa.update(state, action, reward, next_state, next_action)
        state = next_state
        action = next_action  # Important: use the chosen next_action
        total_reward += reward
    
    agent_sarsa.epsilon = max(0.01, agent_sarsa.epsilon * 0.995)  # Decay epsilon
    rewards_sarsa.append(total_reward)
    
    if (e + 1) % 100 == 0:
        print(f"SARSA Episode {e+1}/{num_episodes} - Avg Reward (last 100): {np.mean(rewards_sarsa[-100:]):.2f}")

print(f"Average reward of last 10 episodes for SARSA: {np.mean(rewards_sarsa[-10:]):.2f}")

# Visualize Policies
print("\nPlotting Policies...")
plot_policy(agent_q, env, title="Q-Learning Policy (Optimal Path)")
plot_policy(agent_sarsa, env, title="SARSA Policy (Safer Path)")
