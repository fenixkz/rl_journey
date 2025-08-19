import argparse
import os
import sys

import matplotlib.pyplot as plt
from config.MC_CONFIG import MC_CONFIG
from core.agent import MCAgent  # noqa: E402
from rich import print as pprint
from rich.console import Console

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../")
sys.path.append(os.path.abspath(parent_path))


from common.utils.config_utils import get_env_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MC agent.")
    parser.add_argument(
        "--env",
        type=str,
        default="taxi",
        help="Name of the environment to train on.",
    )
    parser.add_argument("--seed", type=int, default=224)

    args = parser.parse_args()
    return args


def main(args):

    # --- Parse args  ---
    env_name = args.env
    seed = args.seed

    # Get the config for that specific env
    env_config = get_env_config(env_name=env_name, include_atari=False)
    env_id = env_config["env_id"]

    # --- Parse hyperparameters ---
    config = MC_CONFIG
    # Total number of episodes to train
    num_episodes = int(config.get("num_episodes", 1000))
    # Discount rate
    gamma = config.get("gamma", 0.99)
    # Moving average coefficient, or learning rate
    alpha = config.get("alpha", 0.7)
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
    solved_threshold = env_config["solved_reward"]

    # --- Create the agent ---
    agent = MCAgent(
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
    )

    # 1. Create a directory for future results
    save_path = f"results/{env_id}"
    os.makedirs(save_path, exist_ok=True)

    console = Console()
    title = f":rocket: :rocket: :rocket: [bold red] Training {env_id} with Monte-Carlo [/bold red] \
              :rocket: :rocket: :rocket:"
    console.print(title, justify="center")

    # 2. Train the agent
    try:
        # Start training
        agent.train(
            num_episodes=num_episodes,
        )
        pprint("Training completed successfully. Displaying final plot.")
        _ = agent.get_figure()
        plt.show()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
    finally:
        # This will be executed no matter what
        agent.save_progress(save_path=save_path, config=config)


if __name__ == "__main__":
    args = parse_args()
    main(args)
