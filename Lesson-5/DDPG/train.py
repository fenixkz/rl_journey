import argparse
import os
import sys

import matplotlib.pyplot as plt
from config.DDPG_CONFIG import DDPG_CONFIG
from core.agent import DDPGAgent
from rich.console import Console

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../")
sys.path.append(os.path.abspath(parent_path))

from common.utils.config_utils import get_cont_env_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPG agent.")
    parser.add_argument("--env", type=str, default="pendulum", help="Name of the environment to train on.")
    parser.add_argument("--seed", type=int, default=224, help="Random seed for reproducibility.")

    args = parser.parse_args()
    return args


def main(args):

    # --- Parse args and parameters, create env and agent ---
    env_name = args.env
    seed = args.seed

    # Get the config for that specific env
    env_config = get_cont_env_config(env_name=env_name)
    env_id = env_config["env_id"]
    solved_threshold = env_config.get("solved_reward", 100)

    # Some hyperparameters
    config = DDPG_CONFIG
    num_episodes = int(config.get("num_episodes", 10000))
    memory_size = int(config.get("memory_size", int(1e5)))
    initial_noise = config.get("initial_noise", 0.1)
    noise_decay = config.get("noise_decay", 0.99)
    min_noise = config.get("min_noise", 0.01)
    gamma = config.get("gamma", 0.99)
    actor_lr = config.get("actor_lr", 3e-4)
    critic_lr = config.get("critic_lr", 1e-3)
    batch_size = config.get("batch_size", 256)
    tau = config.get("tau", 0.05)
    evaluation_period = config.get("evaluation_period", 100)
    evaluation_episodes = config.get("evaluation_episodes", 10)

    # Create the agent
    agent = DDPGAgent(
        env_id=env_id,
        solved_threshold=solved_threshold,
        seed=seed,
        memory_size=memory_size,
        initial_noise=initial_noise,
        noise_decay=noise_decay,
        min_noise=min_noise,
        gamma=gamma,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        batch_size=batch_size,
        tau=tau,
        evaluation_period=evaluation_period,
        evaluation_episodes=evaluation_episodes,
    )

    # Path where to save the progress
    save_path = f"results/{env_name}"
    os.makedirs(save_path, exist_ok=True)

    console = Console()
    title = (
        f":rocket: :rocket: :rocket: [bold red] Training {env_id}" f" with DDPG [/bold red] :rocket: :rocket: :rocket:"
    )
    console.print(title, justify="center")

    try:
        agent.train(num_episodes=num_episodes)
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
