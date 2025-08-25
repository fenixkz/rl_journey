import argparse
import os
import sys

import matplotlib.pyplot as plt
from config.REINFORCE_CONFIG import REINFORCE_CONFIG
from core.agent import REINFORCEAgent
from rich.console import Console

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../../")
sys.path.append(os.path.abspath(parent_path))

from common.utils.config_utils import get_env_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train a REINFORCE agent.")
    parser.add_argument("--env", type=str, default="cartpole", help="Name of the environment to train on.")
    parser.add_argument("--seed", type=int, default=224)

    args = parser.parse_args()
    return args


def main(args):

    # --- Parse args and parameters, create env and agent ---
    # Parse args
    env_name = args.env
    seed = args.seed

    # Get the config for that specific env
    env_config = get_env_config(env_name=env_name, include_atari=False)
    env_id = env_config["env_id"]
    solved_threshold = env_config.get("solved_reward", 100)

    # Some hyperparameters
    config = REINFORCE_CONFIG
    max_episodes = config.get("max_episodes", 5000)
    learning_rate = config.get("lr", 1e-3)
    hidden_size = config.get("hidden_dim", 128)
    gamma = config.get("gamma", 0.99)
    use_normalization = config.get("use_normalization", False)
    use_entropy = config.get("use_entropy", True)
    entropy_coef = config.get("entropy_coef", 1e-2)
    # Add seed to the config
    config["seed"] = seed

    # Create the agent
    agent = REINFORCEAgent(
        env_id=env_id,
        solved_threshold=solved_threshold,
        gamma=gamma,
        hidden_size=hidden_size,
        lr=learning_rate,
        seed=seed,
        use_normalization=use_normalization,
        use_entropy=use_entropy,
        entropy_coef=entropy_coef,
    )

    # Path where to save the progress
    save_path = f"results/{env_id}"
    os.makedirs(save_path, exist_ok=True)

    console = Console()
    title = (
        f":rocket: :rocket: :rocket: [bold red] Training {env_id}"
        f" with REINFORCE [/bold red] :rocket: :rocket: :rocket:"
    )
    console.print(title, justify="center")

    try:
        agent.train(max_episodes=max_episodes)
        print("\nTraining completed successfully. Displaying final plot.")
        _ = agent.get_figure()
        plt.show()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
    finally:
        agent.save_progress(save_path=save_path, config=config)


if __name__ == "__main__":
    args = parse_args()
    main(args)
