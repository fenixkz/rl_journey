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

from common.utils.classic_config import CLASSIC_ENV_CONFIG  # noqa: E402
from common.utils.config_utils import save_as_json  # noqa: E402
from common.utils.plot_utils import get_figure  # noqa: E402


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


def get_env_config(env_name: str) -> dict:
    """
    Retrieves the configuration for a given environment  name.

    Args:
        env_name: The short name of the environment (e.g., "pong").

    Returns:
        A dictionary containing the environment's configuration.

    Raises:
        ValueError: If the short name is not found in the configs.
    """

    if env_name.lower() in CLASSIC_ENV_CONFIG:
        return CLASSIC_ENV_CONFIG[env_name.lower()]

    raise ValueError(
        f"Unknown environment: '{env_name}'. " f"Available environments: {list(CLASSIC_ENV_CONFIG.keys())}"
    )


def main(args):

    # --- Parse args  ---
    env_name = args.env
    seed = args.seed
    visulize = args.visualize

    # Get the config for that specific env
    env_config = get_env_config(env_name=env_name)
    env_id = env_config["env_id"]

    # --- Parse hyperparameters ---
    config = CEM_CONFIG
    # Total number of learning steps
    training_epochs = config.get("training_epochs", 100)
    # Number of episodes to play
    # for elite filtering
    num_episodes = config.get("num_episodes", 50)
    # Percentile of elite samples to filter
    percentile = config.get("percentile", 60)
    # Size of the hidden layers
    hidden_dim = config.get("hidden_dim", 256)
    # Learning rate
    learning_rate = config.get("lr", 1e-3)
    # Add seed as hyperparam to config
    config["seed"] = seed

    # --- Create the agent ---
    agent = CEMAgent(
        env_id=env_id,
        solved_threshold=env_config["solved_reward"],
        hidden_dim=hidden_dim,
        lr=learning_rate,
        seed=seed,
    )

    # --- Visualization: Untrained Policy ---
    if visulize:
        # Create a similar env but that can visualize the environment using display
        human_env = gym.make(env_id, render_mode="human")

        _, total_reward = agent.play_one_episode(human_env, agent, render=True)
        pprint(f"Total reward achieved by untrained policy [bold red]: {total_reward}")
        human_env.close()

    # Initialize a list of all reward for plotting it later
    all_rewards = []

    # Path where to save the progress
    save_path = f"results/{env_id}"
    os.makedirs(save_path, exist_ok=True)

    # Function to save the progress
    def save_progress():
        """A non-interactive function to save model and plot to file."""
        pprint("\nSaving model and plotting results to file...")
        # 1. Save the agent's policy
        agent.save(save_path)

        # 2. Save config

        save_as_json(save_dir=save_path, config=config)

        # 3. Save the plot of rewards
        if all_rewards:
            fig = get_figure(
                all_rewards=all_rewards,
                solved_threshold=env_config["solved_reward"],
                window_size=num_episodes,
            )
            pprint("Generated the figure")
            save_file_path = os.path.join(save_path, "rewards.jpg")
            try:
                fig.savefig(save_file_path)
                pprint(f"Plot saved to {save_file_path}")
            except Exception as e:
                pprint(f"Could not save plot: {e}")
            finally:
                plt.close(fig)

    console = Console()
    title = f":rocket: :rocket: :rocket: [bold red] Training {env_id} with CEM [/bold red] :rocket: :rocket: :rocket:"
    console.print(title, justify="center")

    training_completed_successfully = False
    try:
        agent.train(
            all_rewards=all_rewards,
            num_epochs=training_epochs,
            num_episodes=num_episodes,
            percentile=percentile,
        )
        training_completed_successfully = True
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
    finally:
        # This block will execute on normal completion, Ctrl+C, or a different error.
        save_progress()

    # Evaluate policy: use deterministic actions
    print("\nEvaluating trained policy...")

    # --- Visualization: Trained Policy ---
    if visulize:
        human_env = gym.make(
            env_id, render_mode="human"
        )  # Create a similar env, but that can visualize the environment using display

        _, total_reward = agent.play_one_episode(env=human_env, deterministic=True, render=True)
        pprint(f"Total reward achieved by trained policy [bold green]: {total_reward}")
        human_env.close()
    # --- End of visualization

    # --- Display the Plot (Only on Normal Completion) ---
    if training_completed_successfully:
        pprint("Training completed successfully. Displaying final plot.")
        # Re-create the figure from the final data and show it.
        _ = get_figure(
            all_rewards=all_rewards,
            solved_threshold=env_config["solved_reward"],
            window_size=num_episodes,
        )
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
