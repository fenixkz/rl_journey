import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from agent import Agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.plot_utils import get_figure

def play_one_episode(env: gym.Env, agent: Agent, render: bool = False):
    """Plays one episode using the provided agent and returns the total reward."""
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.choose_action(state=state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if render: env.render()
        state = next_state
        total_reward += reward
    return total_reward

# --- Environment Configurations ---
ENV_CONFIGS = {
    "CartPole-v1": {
        "training_generations": 100,
        "population_size": 50,
        "noise_std": 0.1,
        "learning_rate": 0.01,
        "threshold_reward": 400,
    },
    "LunarLander-v3": {
        "training_generations": 1000,
        "population_size": 100,
        "noise_std": 0.02,
        "learning_rate": 0.005,
        "threshold_reward": 200,
    },
}

def main():
    # --- Configuration ---
    env_id = "LunarLander-v3"  # "CartPole-v1" "LunarLander-v3"
    # Get a config
    env_config = ENV_CONFIGS.get(env_id, {})

    # ES Hyperparameters
    training_generations = env_config.get("training_generations", 500)  # Number of evolution steps
    population_size = env_config.get("population_size", 50)             # N: Number of "mutants" in a generation
    noise_std = env_config.get("noise_std", 0.01)                       # sigma: The "mutation" strength
    learning_rate = env_config.get("learning_rate", 0.01)               # alpha: How fast the parent evolves
    
    # General Hyperparameters
    threshold_reward = env_config.get("threshold_reward", 100) # LunarLander uses different threshold
    episodes_in_a_row = 0 # Counter for consecutive good episodes
    terminate_after = 2 # Stop training after this many consecutive good episodes

    # --- Setup ---
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Step 1: Start with a "Parent" Policy
    central_agent = Agent(obs_dim, act_dim)
    num_weights = central_agent.get_weights().numel()

    # Performance tracking
    mean_rewards = []
    std_rewards = []
    pbar = tqdm(range(training_generations), desc="Evolving", postfix={"mean_reward": 0})

    for generation in pbar:
        # Get the DNA of the parent
        central_weights = central_agent.get_weights()

        # Step 2: Create a Population of "Mutants"
        # Generate N random mutations
        noise_vectors = [torch.randn(num_weights) for _ in range(population_size)]
        
        rewards = np.zeros(population_size)

        # Step 3: Evaluate the Population's "Fitness"
        for i in range(population_size):
            mutant_agent = Agent(obs_dim, act_dim)
            
            # Create the mutant's DNA by adding noise to the parent's DNA
            mutant_weights = central_weights + noise_std * noise_vectors[i]
            mutant_agent.set_weights(mutant_weights)
            
            # Let the mutant live its life and record its fitness (total reward)
            rewards[i] = play_one_episode(env, mutant_agent)

        # Track performance
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_rewards.append(mean_reward)
        std_rewards.append(std_reward)
        pbar.set_postfix_str(f"mean_reward: {mean_reward:.2f}")

        # Step 4: Natural Selection and Generational Update
        # This is the core update rule: θ_new = θ_old + learning_rate * Σ(R_i * ε_i)
        
        # A common trick: normalize rewards to prevent extreme updates
        normalized_rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        
        # Calculate the weighted sum of mutations
        update_direction = torch.zeros(num_weights)
        for i in range(population_size):
            update_direction += normalized_rewards[i] * noise_vectors[i]
            
        # Evolve the parent's DNA by taking a small step in the successful direction
        # The (population_size * noise_std) term is a standard part of the ES gradient estimator
        update_step = (learning_rate / (population_size * noise_std)) * update_direction
        
        new_weights = central_weights + update_step
        central_agent.set_weights(new_weights)

        # Check for early termination
        if mean_reward > threshold_reward:
            episodes_in_a_row += 1
        else:
            episodes_in_a_row = 0
            
        if episodes_in_a_row >= terminate_after:
            pbar.close()
            print(f"Terminated after {generation+1} steps with mean reward {mean_reward:.2f}")
            break

    # --- Final Evaluation and Visualization ---
    print("\n--- Evaluating Final Evolved Policy ---")
    human_env = gym.make(env_id, render_mode="human")
    final_reward = play_one_episode(human_env, central_agent, render=True)
    print(f"Total reward achieved by final policy: {final_reward}")
    human_env.close()

    save_path = f"results/{env_id}"
    os.makedirs(save_path, exist_ok=True)

    fig = get_figure(mean_rewards, std_rewards, num_episodes=population_size)
    fig.savefig(f"{save_path}/rewards.png")
    plt.show()
    central_agent.save_policy(env_id)
    
    env.close()

if __name__ == "__main__":
    main()
