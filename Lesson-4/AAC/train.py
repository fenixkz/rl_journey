import argparse
import os
import sys

import matplotlib.pyplot as plt
from config.AAC_CONFIG import AAC_CONFIG
from core.agent import AACAgent
from rich.console import Console

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../")
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
    env_config = get_env_config(env_name=env_name, include_atari=False)
    env_id = env_config["env_id"]
    solved_threshold = env_config.get("solved_reward", 100)

    # Some hyperparameters
    config = AAC_CONFIG
    max_steps = int(config.get("max_steps", 1e5))
    actor_lr = config.get("actor_lr", 3e-4)
    critic_lr = config.get("critic_lr", 1e-3)
    gamma = config.get("gamma", 0.9)
    use_normalization = config.get("use_normalization", False)
    use_entropy = config.get("use_entropy", True)
    entropy_coef = config.get("entropy_coef", 1e-2)
    num_steps = config.get("num_steps", 3)
    gae_lambda = config.get("gae_lambda", 0.95)
    hidden_size = config.get("hidden_dim", 128)
    tau = config.get("tau", 0.05)

    # Create the agent
    agent = AACAgent(
        env_id=env_id,
        solved_threshold=solved_threshold,
        gamma=gamma,
        hidden_size=hidden_size,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        seed=seed,
        use_normalization=use_normalization,
        use_entropy=use_entropy,
        entropy_coef=entropy_coef,
        num_steps=num_steps,
        gae_lambda=gae_lambda,
        tau=tau,
    )

    # Path where to save the progress
    save_path = f"results/{env_id}"
    os.makedirs(save_path, exist_ok=True)

    console = Console()
    title = (
        f":rocket: :rocket: :rocket: [bold red] Training {env_id}" f" with AAC [/bold red] :rocket: :rocket: :rocket:"
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
