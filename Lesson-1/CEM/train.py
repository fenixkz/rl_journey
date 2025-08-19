import argparse
import os
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
from config.CEM_CONFIG import CEM_CONFIG
from rich import print as pprint
from rich.console import Console

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../")
sys.path.append(os.path.abspath(parent_path))

from core.agent import CEMAgent  # noqa: E402

from common.utils.config_utils import get_env_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CEM agent.")
    parser.add_argument(
        "--env",
        type=str,
        default="cartpole",
        help="Name of the environment to train on.",
    )
    parser.add_argument("--seed", type=int, default=224)
    parser.add_argument("--visualize", type=bool, default=False)

    args = parser.parse_args()
    return args


def main(args):

    # --- Parse args  ---
    env_name = args.env
    seed = args.seed
    visualize = args.visualize

    # Get the config for that specific env
    env_config = get_env_config(env_name=env_name, include_atari=False)
    env_id = env_config["env_id"]

    # --- Parse hyperparameters ---
    config = CEM_CONFIG
    # Total number of learning steps
    training_epochs = config.get("training_epochs", 100)
    # Number of episodes to play for elite filtering
    num_episodes = config.get("num_episodes", 50)
    # Percentile of elite samples to filter
    percentile = config.get("percentile", 60)
    # Size of the hidden layers
    hidden_dim = config.get("hidden_dim", 256)
    # Learning rate
    learning_rate = config.get("lr", 1e-3)
    # Add seed as hyperparam to config
    config["seed"] = seed
    # A reward after which the env is considered solved
    solved_threshold = env_config["solved_reward"]

    # --- Create the agent ---
    agent = CEMAgent(
        env_id=env_id,
        solved_threshold=solved_threshold,
        hidden_dim=hidden_dim,
        lr=learning_rate,
        seed=seed,
    )

    # --- Visualization: Untrained Policy ---
    if visualize:
        # Create a similar env but that can visualize the environment using display
        human_env = gym.make(env_id, render_mode="human")

        _, total_reward = agent.play_one_episode(human_env, agent, render=True)
        pprint(f"Total reward achieved by untrained policy [bold red]: {total_reward}")
        human_env.close()

    # 1. Create a directory for future results
    save_path = f"results/{env_id}"
    os.makedirs(save_path, exist_ok=True)

    console = Console()
    title = f":rocket: :rocket: :rocket: [bold red] Training {env_id} with CEM [/bold red] :rocket: :rocket: :rocket:"
    console.print(title, justify="center")

    # 2. Train the agent
    try:
        # Start training
        agent.train(
            num_epochs=training_epochs,
            num_episodes=num_episodes,
            percentile=percentile,
        )
        pprint("Training completed successfully. Displaying final plot.")
        # Create the figure from the final data and show it
        _ = agent.get_figure()
        plt.show()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
    finally:
        # Save everything even if the process was killed
        # before the training completed
        agent.save_progress(save_path=save_path, config=config)

    # 3. Evaluate trained policy
    pprint("[green] Evaluating trained policy...")

    # --- Visualization: Trained Policy ---
    if visualize:
        human_env = gym.make(
            env_id, render_mode="human"
        )  # Create a similar env, but that can visualize the environment using display

        _, total_reward = agent.play_one_episode(env=human_env, deterministic=True, render=True)
        pprint(f"Total reward achieved by trained policy [bold green]: {total_reward}")
        human_env.close()
    # --- End of visualization


if __name__ == "__main__":
    args = parse_args()
    main(args)
