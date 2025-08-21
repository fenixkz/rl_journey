import argparse
import os
import sys

import matplotlib.pyplot as plt
from core.agent_registry import registry
from core.configs import DEFAULT_ATARI_PARAMS, DEFAULT_CLASSIC_PARAMS, AgentConfig
from core.DQNBase import DQNBase
from rich.console import Console

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../")
sys.path.append(os.path.abspath(parent_path))

from common.utils.configs import ATARI_CONFIGS, CONFIGS  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL agent on an environment.")
    parser.add_argument("--env", type=str, default="Pong", help="Name of the environment to train on.")
    parser.add_argument(
        "--agent",
        type=int,
        default="1",
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Agent to use for training. \n 1 - DQN \n 2 - Double DQN \n "
        "3 - Double DQN with Prioritized Experience Replay \n 4 - Dueling DQN "
        "\n 5- Dueling DQN with N-step Return and PER \n "
        "6 - Distributional DQN with N-step Return and PER \n"
        "7 - Noisy Net DQN with N-step Return and PER \n 8 - RAINBOW",
    )
    parser.add_argument("--seed", type=int, default=224)
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
    if env_name.lower() in ATARI_CONFIGS:
        return ATARI_CONFIGS[env_name.lower()]

    if env_name.lower() in CONFIGS:
        return CONFIGS[env_name.lower()]

    raise ValueError(
        f"Unknown environment: '{env_name}'. "
        f"Available environments: {list(ATARI_CONFIGS.keys()) + list(CONFIGS.keys())}"
    )


def main(args):
    agent_number = args.agent
    env_config = get_env_config(args.env)

    env_id = env_config["env_id"]
    env_name = env_id.split("/")[-1]

    # Classify whether or not the env is an Atari game
    is_atari = "ALE/" in env_id

    params = DEFAULT_ATARI_PARAMS if is_atari else DEFAULT_CLASSIC_PARAMS
    agent_config = AgentConfig.from_dict(params)
    # Overwrite seed
    agent_config.seed = args.seed
    solved_threshold = env_config["solved_reward"]

    # Create the agent
    agent: DQNBase = registry.create_agent(agent_number, env_id, agent_config, solved_threshold, is_atari)

    save_path = os.path.join("results", agent.get_name(), env_name)
    os.makedirs(save_path, exist_ok=True)

    console = Console()
    title = (
        f":rocket: :rocket: :rocket: [bold red] Training {env_id} "
        f"with {agent.get_name()} [/bold red] :rocket: :rocket: :rocket:"
    )
    console.print(title, justify="center")
    title = f"[bold yellow] Using the following config [/bold yellow]: \n {agent_config}"
    console.print(title, justify="left")

    try:
        agent.train(max_steps=agent_config.max_steps)
        print("\nTraining completed successfully. Displaying final plot.")
        _ = agent.get_figure()
        plt.show()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
    finally:
        agent.save_progress(save_path=save_path, config=agent_config.to_dict())


if __name__ == "__main__":
    args = parse_args()
    main(args)
