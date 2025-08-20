from typing import Tuple, Union

import numpy as np
from core.QLearning import QLearningAgent


class EV_SARSA(QLearningAgent):
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
        super().__init__(
            env_id,
            solved_threshold,
            alpha,
            gamma,
            start_epsilon,
            min_epsilon,
            decay_episodes,
            num_bins,
            evaluation_period,
            evaluation_episodes,
            custom_env,
        )

    def get_ev(self, state):
        """
        Get the action probabilities for the given state.
        This is a epsilon-greedy policy.
        The action with the highest Q-value is selected with probability 1 - epsilon.
        The other actions are selected with probability epsilon / (number of actions).
        """
        q_values = np.array([self.getQ(state, action) for action in range(self.num_actions)])
        max_idx = np.argmax(q_values)
        action_probs = np.ones(self.num_actions) * self.epsilon / self.num_actions
        action_probs[max_idx] += 1 - self.epsilon
        return np.sum(q_values.dot(action_probs))

    def learn(
        self,
        state: Union[np.ndarray, Tuple[int, ...]],
        action: int,
        reward: float,
        next_state: Union[np.ndarray, Tuple[int, ...]],
    ):
        """
        SARSA update rule:
            TD Target = reward + gamma * E[V(s')]
        """
        current_q = self.getQ(state, action)
        next_V = self.get_ev(next_state)
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_V)
        self.setQ(state, action, new_q)
