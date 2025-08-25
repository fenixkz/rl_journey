import argparse
import os
import sys
import time

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
from core.agent_registry import registry
from core.configs import AgentConfig
from core.DQNBase import DQNBase

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../")
sys.path.append(os.path.abspath(parent_path))

from common.utils.atari_utils import get_atari_env  # noqa: E402
from common.utils.config_utils import get_env_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a trained policy.")

    parser.add_argument("--env", type=str, default="cartpole", help="Name of the environment to visualize.")
    parser.add_argument("--gif", action="store_true", help="Generate a GIF of the episode.")
    parser.add_argument(
        "--agent",
        type=int,
        default="1",
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Agent to use for evaluation. \n 1 - DQN \n 2 - Double DQN \n "
        "3 - Double DQN with Prioritized Experience Replay \n 4 - Dueling DQN "
        "\n 5- Dueling DQN with N-step Return and PER \n "
        "6 - Distributional DQN with N-step Return and PER \n"
        "7 - Noisy Net DQN with N-step Return and PER \n 8 - RAINBOW",
    )
    args = parser.parse_args()
    return args


def main(args):
    # --- Parse args  ---
    env_name = args.env
    generate_gif = args.gif
    agent_number = args.agent

    # Get the config for that specific env
    env_config = get_env_config(env_name=env_name)
    env_id = env_config["env_id"]
    env_name = env_id.split("/")[-1]

    # Is Atari or not
    is_atari = "ALE/" in env_id

    if is_atari:
        env = get_atari_env(env_id)
    else:
        env = gym.make(env_id, render_mode="rgb_array")

    agent: DQNBase = registry.create_agent(agent_number, env_id, AgentConfig, {}, is_atari)

    # Check if the env was trained with this agent
    load_path = os.path.join("results", agent.name, env_name)
    if not os.path.exists(load_path):
        print(f"Error env: {env_name} was not trained with the agent {agent.name}")
        return
    # Re-create the agent with the loaded config from json file
    agent: DQNBase = registry.create_agent(
        agent_number, env_id, AgentConfig.from_json(os.path.join(load_path, "config.json")), {}, is_atari
    )
    agent.load(load_path=load_path)

    frames_for_gif = [] if generate_gif else None

    # Reset the environment to get the first observation
    observation, _ = env.reset()

    # 2. Set up the Matplotlib plot for interactive display
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    # Render the first frame and display it in the plot
    frame = env.render()
    img = ax.imshow(frame)
    plt.axis("off")  # Hide the axes for a cleaner look

    print("Starting visualization. Press Ctrl+C in the terminal to stop.")
    total_reward = 0
    try:
        while True:
            # --- This is where you would insert your trained agent's policy ---
            action = agent.choose_action(observation, epsilon=0.0)

            # Step through the environment
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # 3. Render the new frame and update the plot data
            frame = env.render()
            img.set_data(frame)

            if generate_gif:
                frames_for_gif.append(frame)

            # Redraw the canvas
            fig.canvas.draw()
            fig.canvas.flush_events()

            # A small pause is needed to allow the GUI to update and to control the speed
            time.sleep(0.02)

            # If the episode has ended, reset to start a new one
            if terminated or truncated:
                print(f"Total evaluation reward: {total_reward}")
                break
    except KeyboardInterrupt:
        print("Visualization stopped by user.")
    finally:
        # 4. Clean up
        env.close()
        plt.ioff()  # Turn off interactive mode
        print("Environment closed.")

    if generate_gif and frames_for_gif:
        print("Generating GIF...")
        # Create a path to save the GIF inside the agent's results folder
        gif_path = os.path.join("results", agent.name, env_name)
        os.makedirs(gif_path, exist_ok=True)
        filename = os.path.join(gif_path, f"{env_name}.gif")

        # Use imageio to save the frames at 30 FPS
        imageio.mimsave(filename, frames_for_gif, fps=30)
        print(f"✅ GIF saved successfully to {filename}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
