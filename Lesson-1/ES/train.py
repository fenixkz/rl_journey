import argparse
import multiprocessing as mp
import os
import sys

import matplotlib.pyplot as plt
from config.ES_CONFIG import ES_ATARI_CONFIG, ES_CONFIG
from core.agent import ESAgent
from rich import print as pprint
from rich.console import Console

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../")
sys.path.append(os.path.abspath(parent_path))

from common.utils.config_utils import get_env_config  # noqa: E402
from common.utils.general_utils import save_progress  # noqa: E402
from common.utils.plot_utils import get_figure  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ES agent.")
    parser.add_argument("--env", type=str, default="cartpole", help="Name of the environment to train on.")
    parser.add_argument("--seed", type=int, default=224)
    parser.add_argument(
        "--cpu-usage",
        type=float,
        default=0.5,
        help="Part of total CPU cores to use for training for multi-processing, "
        "1 - all cores, 0.1 - 10 percent. Only used for Atari envs.",
    )

    args = parser.parse_args()
    return args


def main(args):

    # --- Parse args  ---
    env_name = args.env
    seed = args.seed
    cpu_usage = max(args.cpu_usage, 0.1)

    # Get the config for that specific env
    env_config = get_env_config(env_name=env_name)
    env_id = env_config["env_id"]

    # --- ES hyperparameters ---
    # A flag to check whether this is atari or not
    is_atari = "ALE/" in env_id
    config = ES_ATARI_CONFIG if is_atari else ES_CONFIG
    training_generations = config.get("training_generations", 100)  # Number of evolution steps
    population_size = config.get("population_size", 50)  # N: Number of "mutants" in a generation
    noise_std = config.get("noise_std", 0.01)  # sigma: The "mutation" strength
    hidden_dim = config.get("hidden_dim", 256)  # size of hidden layers
    learning_rate = config.get("learning_rate", 0.01)  # alpha: How fast the parent evolves
    use_vbn = config.get("use_vbn", True)  # whether or not use Virtual Batch Norm (VBN)
    vbn_batch_size = config.get("vbn_batch_size", 128)  # a batch size to calculate state for VBN
    l2_coeff = config.get("l2_coeff", 0.005)
    normalization_mode = config.get("normalization_mode", "default")
    action_noise = config.get("action_noise", 0.1)
    evaluation_period = config.get("evaluation_period", 100)
    evaluation_episodes = config.get("evaluation_episodes", 50)
    solved_threshold = env_config["solved_reward"]

    # --- Create ES agent ---
    agent = ESAgent(
        env_id=env_id,
        solved_threshold=solved_threshold,
        noise_std=noise_std,
        is_atari=is_atari,
        hidden_dim=hidden_dim,
        seed=seed,
        learning_rate=learning_rate,
        cpu_usage=cpu_usage,
        use_vbn=use_vbn,
        vbn_batch_size=vbn_batch_size,
        calculate_vbn_params=True,
        l2_coeff=l2_coeff,
        normalization_mode=normalization_mode,
        action_noise=action_noise,
        evaluation_period=evaluation_period,
        evaluation_episodes=evaluation_episodes,
    )

    # 1. Create a directory for future results
    save_path = f"results/{env_id}"
    os.makedirs(save_path, exist_ok=True)

    console = Console()
    title = f":rocket: :rocket: :rocket: [bold red] Training {env_id} with ES [/bold red] :rocket: :rocket: :rocket:"
    console.print(title, justify="center")

    training_completed_successfully = False
    try:
        # A collection of rewards for future plot
        train_rewards, eval_rewards = [], []

        # Start training
        agent.train(
            train_rewards=train_rewards,
            num_epochs=training_generations,
            population_size=population_size,
            eval_rewards=eval_rewards,
        )
        training_completed_successfully = True
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
    finally:
        # This block will execute on normal completion, Ctrl+C, or a different error.
        save_progress(
            agent=agent,
            save_path=save_path,
            config=config,
            train_rewards=train_rewards,
            window_size=100,
            solved_threshold=solved_threshold,
            eval_rewards=eval_rewards,
        )

    # --- Display the Plot (Only on Normal Completion) ---
    if training_completed_successfully:
        pprint("Training completed successfully. Displaying final plot.")
        # Re-create the figure from the final data and show it.
        _ = get_figure(
            train_rewards=train_rewards,
            solved_threshold=env_config["solved_reward"],
            window_size=population_size,
            eval_rewards=eval_rewards,
        )
        plt.show()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # The start method can only be set once.
    args = parse_args()
    main(args)
