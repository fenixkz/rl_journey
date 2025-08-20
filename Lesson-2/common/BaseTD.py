import os
import sys
from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../")
sys.path.append(os.path.abspath(parent_path))

from common.base_agent import BaseAgent  # noqa: E402


class TDLearningAgent(BaseAgent):
    def __init__(
        self,
        env_id: str,
        solved_threshold: float,
        alpha: float = 0.1,
        gamma: float = 0.9,
        start_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        decay_episodes: int = 100000,
        num_bins: int = 100,
        evaluation_period: int = 100,
        evaluation_episodes: int = 50,
        custom_env=None,
    ):
        # Create the environment using gymnasium
        if custom_env is None:
            env = gym.make(env_id)
            assert isinstance(
                env.action_space, gym.spaces.Discrete
            ), "Detected non-discrete action space, this class works only with discrete action space problems!"
        else:
            # Or use our custom WindyBridge env
            env = custom_env
        super().__init__(env=env, solved_threshold=solved_threshold)
        self.alpha = alpha
        self.gamma = gamma
        self.start_epsilon = start_epsilon
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.decay_episodes = decay_episodes
        # Calculate decay rate: after decay_episodes, epsilon should reach min_epsilon
        self.decay_rate = (min_epsilon / start_epsilon) ** (1 / decay_episodes)
        self.current_episode = 0
        self.action_space = env.action_space
        self.num_actions = env.action_space.n
        self.evaluation_episodes = evaluation_episodes
        self.evaluation_period = evaluation_period
        # Our Q-values are stored in a dictionary
        # that's why we can apply it to a small set of problems
        # with low state space
        self._Q = {}

        # Alternatively, we can discretize the state space
        # --- Discretization Setup ---
        self.is_continuous = isinstance(env.observation_space, gym.spaces.Box)
        if self.is_continuous:
            print("Environment is continuous, applying discrete logic")
            self.num_bins = num_bins
            self.state_bins = self._discretize_space(env.observation_space)

    def _discretize_space(self, obs_space: gym.spaces.Box) -> List[np.ndarray]:
        """Create bins for each dimension of the continuous observation space."""
        state_bins = []
        for i in range(obs_space.shape[0]):
            low = obs_space.low[i]
            high = obs_space.high[i]
            if low == -np.inf:
                low = -5.0
            if high == np.inf:
                high = 5.0
            # Create a list of bin edges for this dimension
            bins = np.linspace(low, high, self.num_bins + 1)[1:-1]
            state_bins.append(bins)
        return state_bins

    def _discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """Convert a continuous state vector into a discrete tuple of bin indices."""
        # If the environment is already discrete, no need to do anything
        if not self.is_continuous:
            return state

        binned_state = []
        for i, val in enumerate(state):
            # Find which bin the value falls into for this dimension
            bin_index = np.digitize(val, self.state_bins[i])
            binned_state.append(bin_index)
        return tuple(binned_state)

    def decay_epsilon(self):
        """
        Decay epsilon using exponential decay over decay_episodes.
        After decay_episodes episodes, epsilon will reach min_epsilon.
        """
        self.current_episode += 1
        if self.current_episode < self.decay_episodes:
            self.epsilon = max(self.min_epsilon, self.start_epsilon * (self.decay_rate**self.current_episode))
        else:
            self.epsilon = self.min_epsilon

    def getQ(self, state, action):
        return self._Q.get((state, action), 0.0)

    def setQ(self, state, action, q):
        self._Q[(state, action)] = q

    def choose_action(self, state: Union[np.ndarray, Tuple[int, ...]], epsilon: Optional[float] = None):
        """
        Use epsilon-greedy policy to choose an action
        """
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.random() < epsilon:
            return self.action_space.sample()
        else:
            q_values = [self.getQ(state, action) for action in range(self.num_actions)]
            return np.argmax(q_values)

    def learn(
        self,
        state: Union[np.ndarray, Tuple[int, ...]],
        action: int,
        reward: float,
        next_state: Union[np.ndarray, Tuple[int, ...]],
        next_action: Optional[int] = None,
    ):
        raise NotImplementedError

    def play_one_episode(self, epsilon: Optional[float] = None, validation: bool = False):
        """
        Play a complete episode in the environment using the current policy.
        """
        raise NotImplementedError

    def train(self, num_episodes: int):
        """ """
        raise NotImplementedError

    def save(self, save_path: str):
        print("--- Saving Policy ---")
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(f"{save_path}/policy.npz", q_table=self._Q)
        print(f"Policy saved in {save_path}")

    def load(self, load_path: str):
        try:
            # Load the .npz file
            data = np.load(f"{load_path}/policy.npz", allow_pickle=True)
            # Extract the dictionary using its key and .item()
            self._Q = data["q_table"].item()
            print(f"Policy loaded from {load_path}")
        except FileNotFoundError:
            print(f"Policy file not found in {load_path}. Using untrained policy.")
