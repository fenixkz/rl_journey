import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from config.TD_CONFIG import TD_CONFIG
from core.EV_SARSA import EV_SARSA
from core.QLearning import QLearningAgent
from core.SARSA import SARSA

from utils.windy_env import WindyBridgeEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TD agent.")
    parser.add_argument("--seed", type=int, default=224)

    args = parser.parse_args()
    return args


def visualize_policy(agent: QLearningAgent, env: WindyBridgeEnv, title: str = "Agent Policy"):
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
        0: (0, -1),  # Up
        1: (1, 0),  # Right
        2: (0, 1),  # Down
        3: (-1, 0),  # Left
    }

    # Populate the policy grid
    for r in range(rows):
        for c in range(cols):
            state = env._location_to_state((r, c))
            # Don't draw arrows in terminal states
            if np.array_equal((r, c), env.goal_pos) or (r, c) in env.chasm_coords:
                continue

            # Get the best action from the agent's policy
            action = agent.choose_action(state, epsilon=0.0)

            if action in action_vectors:
                dx, dy = action_vectors[action]
                u[r, c] = dx
                v[r, c] = -dy  # Invert y for plotting

    # --- Draw the environment background ---
    # Goal (Green)
    ax.add_patch(plt.Rectangle((env.goal_pos[1] - 0.5, env.goal_pos[0] - 0.5), 1, 1, color="green", alpha=0.5))
    # Bridge (Tan)
    for pos in env.bridge_coords:
        ax.add_patch(plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, color="tan"))
    # Chasm (Blue)
    for pos in env.chasm_coords:
        ax.add_patch(plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, color="darkblue", alpha=0.7))

    # --- Draw the policy arrows ---
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    ax.quiver(x, y, u, v, color="black", headwidth=4, headlength=5, scale=25)

    # --- Formatting ---
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()  # Match the (row, col) coordinate system
    ax.set_title(title)

    return fig


def main(args):
    # --- Parse args  ---
    seed = args.seed

    # Get the config for that specific env

    # --- Parse hyperparameters ---
    config = TD_CONFIG
    # Total number of episodes to train
    num_episodes = int(config.get("num_episodes", 1000))
    # Discount rate
    gamma = config.get("gamma", 0.99)
    # Moving average coefficient, or learning rate
    alpha = config.get("alpha", 0.1)
    # Starting epsilon
    start_epsilon = config.get("start_epsilon", 1.0)
    # Finishing epsilon
    min_epsilon = config.get("min_epsilon", 0.1)
    # Number of episodes to decay epsilon from start to min
    decay_episodes = int(config.get("decay_episodes", 100000))
    # Number of bins for continuous state spaces
    num_bins = config.get("num_bins", 100)
    # Evaluation period
    evaluation_period = config.get("evaluation_period", 200)
    # Evaluation episodes
    evaluation_episodes = config.get("evaluation_episodes", 100)
    # Add seed as hyperparam to config
    config["seed"] = seed
    # A reward after which the env is considered solved
    solved_threshold = 1000

    # --- Initialize our custom env ---
    env_id = "WindyBridge-v0"
    env = WindyBridgeEnv()

    # --- Q-Learning ---

    q_learning_agent = QLearningAgent(
        env_id=env_id,
        solved_threshold=solved_threshold,
        alpha=alpha,
        gamma=gamma,
        start_epsilon=start_epsilon,
        min_epsilon=min_epsilon,
        decay_episodes=decay_episodes,
        num_bins=num_bins,
        evaluation_period=evaluation_period,
        evaluation_episodes=evaluation_episodes,
        custom_env=env,
    )

    q_learning_agent.train(num_episodes=num_episodes)
    q_learning_policy = visualize_policy(q_learning_agent, env, title="Q-Learning Policy")

    os.makedirs("results/q-learning/WindyBridge", exist_ok=True)
    q_learning_policy.savefig("results/q-learning/WindyBridge/policy.jpg")
    plt.close(q_learning_policy)

    # --- SARSA ----

    sarsa_agent = SARSA(
        env_id=env_id,
        solved_threshold=solved_threshold,
        alpha=alpha,
        gamma=gamma,
        start_epsilon=start_epsilon,
        min_epsilon=min_epsilon,
        decay_episodes=decay_episodes,
        num_bins=num_bins,
        evaluation_period=evaluation_period,
        evaluation_episodes=evaluation_episodes,
        custom_env=env,
    )

    sarsa_agent.train(num_episodes=num_episodes)
    sarsa_policy = visualize_policy(sarsa_agent, env, title="SARSA Policy")
    os.makedirs("results/sarsa/WindyBridge", exist_ok=True)
    sarsa_policy.savefig("results/sarsa/WindyBridge/policy.jpg")
    plt.close(sarsa_policy)

    # --- EV-SARSA ---

    ev_sarsa_agent = EV_SARSA(
        env_id=env_id,
        solved_threshold=solved_threshold,
        alpha=alpha,
        gamma=gamma,
        start_epsilon=start_epsilon,
        min_epsilon=min_epsilon,
        decay_episodes=decay_episodes,
        num_bins=num_bins,
        evaluation_period=evaluation_period,
        evaluation_episodes=evaluation_episodes,
        custom_env=env,
    )

    ev_sarsa_agent.train(num_episodes=num_episodes)
    ev_sarsa_policy = visualize_policy(ev_sarsa_agent, env, title="EV-SARSA Policy")
    os.makedirs("results/ev-sarsa/WindyBridge", exist_ok=True)
    ev_sarsa_policy.savefig("results/ev-sarsa/WindyBridge/policy.jpg")
    plt.close(ev_sarsa_policy)


if __name__ == "__main__":
    args = parse_args()
    main(args)
