from typing import Optional, Tuple, Union

import numpy as np
from core.QLearning import QLearningAgent


class SARSA(QLearningAgent):
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

    def learn(
        self,
        state: Union[np.ndarray, Tuple[int, ...]],
        action: int,
        reward: float,
        next_state: Union[np.ndarray, Tuple[int, ...]],
        next_action: int,
    ):
        """
        SARSA update rule:
            TD Target = reward + gamma * Q(s_{t+1}, a_{t+1})
        """
        current_q = self.getQ(state, action)
        next_q = self.getQ(next_state, next_action)
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_q)
        self.setQ(state, action, new_q)

    def play_one_episode(self, epsilon: Optional[float] = None, validation: bool = False):
        """
        Play a complete episode in the environment using the current policy.
        """
        # Get initial state, note no seed is needed as it was seeded in the base class
        state, _ = self.env.reset()
        # A flag whether the episode has finished
        done = False
        # Variable to track the total episode's reward
        total_reward = 0
        next_action = None
        # Loop while it is still not done
        while not done:
            # Use discrete version of a state if continuos space
            state = self._discretize_state(state)
            # Get action from our policy
            if next_action is None:
                action = self.choose_action(state=state, epsilon=epsilon)
            else:
                action = next_action
            # Apply our action and receive the feedback
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            # Was the step terminal?
            done = terminated or truncated
            if not validation:
                next_action = self.choose_action(state=self._discretize_state(next_state))
                self.learn(state, action, reward, self._discretize_state(next_state), next_action)
            # Transit to the next observation
            state = next_state
            total_reward += reward

        return total_reward
