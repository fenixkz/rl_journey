import argparse
import os
import sys
import time

import gymnasium as gym
from core.agent import SACAgent

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../../")
sys.path.append(os.path.abspath(parent_path))

from common.utils.config_utils import get_cont_env_config  # noqa: E402


def visualize_environment(agent: SACAgent, env_id: str):

    print(f"--- Visualizing Environment: {env_id} ---")
    env = gym.make(env_id, render_mode="human")

    # Get the recommended rendering frame rate from the environment's metadata
    render_fps = env.metadata.get("render_fps", 60)

    # Reset the environment to get the initial state
    observation, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Take a random action from the environment's action space
        action = agent.choose_action(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Control the rendering speed to make it watchable
        time.sleep(1 / render_fps)
        total_reward += reward

    # Clean up the environment resources
    env.close()
    print(f"Total reward: {total_reward}")
    print("--- Visualization Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize an environment.")
    parser.add_argument("--env", type=str, default="pendulum", help="Environment name.")

    args = parser.parse_args()

    env_config = get_cont_env_config(args.env)
    env_id = env_config["env_id"]
    solved_threshold = env_config.get("solved_reward", 100)

    agent = SACAgent(env_id=env_id, solved_threshold=solved_threshold)

    load_path = os.path.join("results", args.env)
    try:
        agent.load(load_path=load_path)
    except FileNotFoundError:
        raise FileNotFoundError("Saved models were not found")

    visualize_environment(agent, env_id)
