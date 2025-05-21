import gymnasium as gym
import numpy as np
from typing import Optional, Any, Tuple, List, Dict, Union

# ===============================================================
# Paste your MyTrainingEnv class definition here
# ===============================================================
class MyTrainingEnv:
    """
    A wrapper around a Gymnasium environment to add custom methods
    or attributes for training.
    """
    def __init__(self, name, render_mode=None, **kwargs):
        # Added error handling for gym.make
        try:
            self.env = gym.make(name, render_mode=render_mode, **kwargs)
            self.spec = self.env.spec # Store spec for later use if needed
        except gym.error.Error as e:
            print(f"Error creating environment '{name}': {e}")
            raise # Re-raise the exception

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._render_mode = render_mode

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        # Ensure seed is passed down correctly
        # Note: Gymnasium >0.26 prefers seeding via reset, <0.26 sometimes via make
        try:
            # Pass seed directly to the underlying env's reset
            return self.env.reset(seed=seed, options=options)
        except TypeError as e:
            # Fallback if the underlying env reset doesn't accept seed (older style?)
            # You might need to handle seeding differently based on env version
            print(f"Warning: Environment reset might not support seeding directly: {e}")
            return self.env.reset(options=options)


    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def run_episode(self, agent: Any, n_steps: Optional[int] = None, deterministic: bool = False) -> Tuple[List, List, float]:
        """Runs a single episode within this specific environment instance."""
        states = []
        actions = []
        rewards_sum = 0.0
        # Determine max steps safely
        max_episode_steps = 1000 # Default fallback
        if hasattr(self, 'spec') and self.spec and self.spec.max_episode_steps:
            max_episode_steps = self.spec.max_episode_steps
        n_steps = n_steps or max_episode_steps

        try:
            state, info = self.reset() # Use internal reset
        except Exception as e:
            print(f"Error during reset in run_episode: {e}")
            raise

        done = False
        truncated = False
        count = 0
        while not (done or truncated) and count < n_steps:
            try:
                action = agent.act(state, deterministic) # Agent acts on single state
            except Exception as e:
                print(f"Error during agent action in run_episode: {e}")
                raise

            try:
                next_state, reward, done, truncated, info = self.step(action)
                # print(f"Debug: step done, reward: {reward}, done: {done}, trunc: {truncated}") # Debug print
            except Exception as e:
                print(f"Error during environment step in run_episode: {e}")
                raise

            states.append(state) # Store state *before* step
            actions.append(action)
            rewards_sum += reward
            state = next_state # Update state for next iteration
            count += 1

        return states, actions, rewards_sum

    def collect_data(self, n_episodes: int = 10, agent: Any = None) -> Tuple[List, List, List]:
        """Collects data by running multiple episodes in this single env."""
        all_states = []
        all_actions = []
        all_rewards_sum_per_episode = [] # Store total reward per episode
        for i in range(n_episodes):
            # print(f"Debug: Starting episode {i+1}/{n_episodes}") # Debug print
            try:
                s, a, r_sum = self.run_episode(agent)
                # print(f"Debug: Finished episode {i+1}, reward: {r_sum}, states: {len(s)}") # Debug print
                all_states.append(s) # Flatten lists for easier processing later? Or keep per episode? Depends on need.
                all_actions.append(a)
                all_rewards_sum_per_episode.append(r_sum)
            except Exception as e:
                print(f"Error running episode {i+1} in collect_data: {e}")
                raise
        # Returns flattened states/actions and list of episode rewards
        return all_states, all_actions, all_rewards_sum_per_episode


    @property
    def underlying_env(self):
        return self.env
# ===============================================================

# ===============================================================
# Paste your make_my_wrapped_env function definition here
# ===============================================================
def make_my_wrapped_env(env_id, seed, render_mode=None, wrapper_config=None):
    """Returns a function that creates an instance of MyTrainingEnv."""
    def _init():
        # This inner function is what AsyncVectorEnv needs
        # print(f"  _init: Creating MyTrainingEnv '{env_id}' with seed {seed}") # Debug print
        env = MyTrainingEnv(env_id, render_mode=render_mode)
        # We pass the seed to the reset method now, as per Gymnasium standard
        # env.reset(seed=seed) # Reset is called by AsyncVectorEnv automatically initially
        # We rely on AsyncVectorEnv calling reset with the correct seed later.
        return env
    return _init
# ===============================================================


# --- Function to construct the Vectorized Environment ---

def create_async_vector_env(env_id: str, num_envs: int, base_seed: int = 0):
    """
    Creates and returns an AsyncVectorEnv running instances of MyTrainingEnv.
    """
    if num_envs <= 0:
        print("Error: num_envs must be positive.")
        return None

    print(f"\n--- Creating Vectorized Environment ---")
    print(f"Base Environment ID: {env_id}")
    print(f"Number of Parallel Workers: {num_envs}")
    print(f"Base Seed: {base_seed}")

    # Create a list of the actual init functions, assigning unique seeds
    env_fns = []
    for i in range(num_envs):
        seed = base_seed + i
        env_fns.append(make_my_wrapped_env(env_id, seed)) # Pass seed here for make_env logic

    try:
        # Create the AsyncVectorEnv
        vec_env = gym.vector.AsyncVectorEnv(env_fns)
        print(f"\nSuccessfully created AsyncVectorEnv.")
        # Print some info about the vectorized env
        print(f"  Number of environments: {vec_env.num_envs}")
        print(f"  Observation Space: {vec_env.observation_space}")
        print(f"  Action Space: {vec_env.action_space}")
        return vec_env

    except Exception as e:
        print(f"\nError creating AsyncVectorEnv: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        print("Please ensure:")
        print(f"  - Environment ID '{env_id}' is correct.")
        print(f"  - Required dependencies for '{env_id}' are installed.")
        print(f"  - Your MyTrainingEnv wrapper initializes correctly.")
        return None

def collect_vectorized_data(
    vec_env: gym.vector.VectorEnv,
    agent: Any, # Your agent instance (e.g., RandomAgent or a real one)
    num_steps: int # The number of steps *per environment* to collect
) -> Dict[str, Union[np.ndarray, List]]:
    """
    Collects trajectories from a vectorized environment for a specific number of steps.

    Args:
        vec_env: The vectorized Gymnasium environment.
        agent: An agent object with an `act(observations)` method that returns batched actions.
        num_steps: The number of steps to run *in each parallel environment*.
                 The total number of transitions collected will be num_envs * num_steps.

    Returns:
        A dictionary containing the collected experience, e.g.,
        {'observations': ..., 'actions': ..., 'rewards': ..., 'next_observations': ..., 'dones': ...}
        Each value is typically a NumPy array or list covering all steps from all envs.
    """
    print(f"\n--- Collecting Vectorized Data ---")
    print(f"Number of steps per environment: {num_steps}")
    num_envs = vec_env.num_envs
    total_transitions = num_envs * num_steps
    print(f"Total transitions to collect: {total_transitions}")

    # Initialize storage for the trajectories
    # Store the observation *before* the step
    all_obs = [None] * total_transitions
    all_actions = [None] * total_transitions
    all_rewards = [None] * total_transitions
    all_next_obs = [None] * total_transitions
    all_dones = [None] * total_transitions
    # You might want to store infos too, especially 'final_observation'
    # all_infos = [] # Could store the full info dicts

    # Get the initial state for all environments
    # Use a temporary variable for the current observation in the loop
    current_obs, infos = vec_env.reset()
    print(f"Initial observation batch shape: {current_obs.shape}")

    # Counter for stored transitions
    storage_idx = 0

    # Main collection loop
    for step in range(num_steps):
        # Get actions from the agent (handling the batch)
        actions = agent.act(current_obs)

        # Store the observation that led to this action
        # Store actions
        for i in range(num_envs):
             # Calculate the correct index in the flattened storage
            current_storage_idx = storage_idx + i
            if current_storage_idx < total_transitions:
                all_obs[current_storage_idx] = current_obs[i]
                all_actions[current_storage_idx] = actions[i]

        # Step the environments
        next_obs, rewards, terminations, truncations, infos = vec_env.step(actions)

        # Store rewards, next_obs, and dones
        for i in range(num_envs):
             # Calculate the correct index in the flattened storage
            current_storage_idx = storage_idx + i
            if current_storage_idx < total_transitions:
                all_rewards[current_storage_idx] = rewards[i]
                all_next_obs[current_storage_idx] = next_obs[i]
                # Combine done flags
                all_dones[current_storage_idx] = terminations[i] or truncations[i]

                # Handle final observations if an env terminated/truncated
                # This is important for calculating value targets correctly
                if terminations[i] or truncations[i]:
                    # If an env is done, the 'next_obs' is actually the reset obs.
                    # The 'infos' dict usually contains the 'final_observation'.
                    if 'final_observation' in infos:
                         # Use numpy array access to get env-specific info
                        final_obs_for_env_i = infos['final_observation'][i]
                        if final_obs_for_env_i is not None:
                            # Store the actual final observation before reset
                            all_next_obs[current_storage_idx] = final_obs_for_env_i
                            # print(f"Debug: Stored final_observation for env {i} at index {current_storage_idx}") # Debug

        # Update the current observation for the next loop iteration
        current_obs = next_obs
        # Increment storage index by the number of parallel environments
        storage_idx += num_envs


    print(f"Data collection complete. Stored {storage_idx} transitions.")

    # Prepare the data dictionary (convert lists to numpy arrays)
    # Note: Handle potential None values if collection was interrupted (though loop should prevent it)
    collected_data = {
        "observations": np.array(all_obs, dtype=vec_env.observation_space.dtype),
        "actions": np.array(all_actions), # Dtype depends on action space
        "rewards": np.array(all_rewards, dtype=np.float32),
        "next_observations": np.array(all_next_obs, dtype=vec_env.observation_space.dtype),
        "dones": np.array(all_dones, dtype=np.bool_)
    }

    # You might want to reshape arrays here depending on your needs,
    # e.g., shape (num_envs * num_steps, *feature_shape) or (num_steps, num_envs, *feature_shape)

    return collected_data


