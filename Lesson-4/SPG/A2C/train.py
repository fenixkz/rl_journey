import argparse
import os
import sys

import matplotlib.pyplot as plt
from config.A2C_CONFIG import A2C_CONFIG
from core.agent import A2CAgent
from rich.console import Console

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../../")
sys.path.append(os.path.abspath(parent_path))

from common.utils.config_utils import get_env_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train a AAC agent with comprehensive debugging.")
    parser.add_argument("--env", type=str, default="cartpole", help="Name of the environment to train on.")
    parser.add_argument("--seed", type=int, default=224, help="Random seed for reproducibility.")

    args = parser.parse_args()
    return args


def main(args):

    # --- Parse args and parameters, create env and agent ---
    env_name = args.env
    seed = args.seed

    # Get the config for that specific env
    env_config = get_env_config(env_name=env_name, include_atari=True)
    env_id = env_config["env_id"]
    env_name = env_id.split("/")[-1]
    # Classify whether or not the env is an Atari game
    is_atari = "ALE/" in env_id
    solved_threshold = env_config.get("solved_reward", 100)

    # Some hyperparameters
    config = A2C_CONFIG
    num_envs = config.get("num_envs", 5)
    max_steps = int(config.get("max_steps", 1e5))
    lr = config.get("lr", 3e-4)
    gamma = config.get("gamma", 0.9)
    use_normalization = config.get("use_normalization", False)
    use_entropy = config.get("use_entropy", True)
    entropy_coef = config.get("entropy_coef", 1e-2)
    num_steps = config.get("num_steps", 3)
    gae_lambda = config.get("gae_lambda", 0.95)
    hidden_size = config.get("hidden_dim", 128)
    tau = config.get("tau", 0.05)
    critic_scale = config.get("critic_scale", 0.5)

    # Create the agent
    agent = A2CAgent(
        env_id=env_id,
        num_envs=num_envs,
        solved_threshold=solved_threshold,
        critic_scale=critic_scale,
        is_atari=is_atari,
        gamma=gamma,
        hidden_size=hidden_size,
        lr=lr,
        seed=seed,
        use_normalization=use_normalization,
        use_entropy=use_entropy,
        entropy_coef=entropy_coef,
        num_steps=num_steps,
        gae_lambda=gae_lambda,
        tau=tau,
    )

    # Path where to save the progress
    save_path = f"results/{env_name}"
    os.makedirs(save_path, exist_ok=True)

    console = Console()
    title = (
        f":rocket: :rocket: :rocket: [bold red] Training {env_id}" f" with A2C [/bold red] :rocket: :rocket: :rocket:"
    )
    console.print(title, justify="center")

    try:
        agent.train(max_steps=max_steps)
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
