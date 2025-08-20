import os
import sys
from typing import Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../")
sys.path.append(os.path.abspath(parent_path))

from common.BaseTD import TDLearningAgent  # noqa: E402


class QLearningAgent(TDLearningAgent):

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
    ):
        """
        Q-learning update rule, compute TD target as (r_t + max{Q(s_{t+1}, a)})
        """
        current_q = self.getQ(state, action)
        next_max = max([self.getQ(next_state, a) for a in range(self.num_actions)])
        new_q = current_q + self.alpha * (reward + self.gamma * next_max - current_q)
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

        # Loop while it is still not done
        while not done:
            # Use discrete version of a state if continuos space
            state = self._discretize_state(state)
            # Get action from our policy
            action = self.choose_action(state=state, epsilon=epsilon)
            # Apply our action and receive the feedback
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            # Was the step terminal?
            done = terminated or truncated
            if not validation:
                # Next state should be discrete as well
                self.learn(state, action, reward, self._discretize_state(next_state))
            # Transit to the next observation
            # Note next_state is continuos (if is_continuos)
            # So, state will be also continuous
            state = next_state
            total_reward += reward

        return total_reward

    def train(self, num_episodes: int):
        """ """

        pbar = tqdm(range(num_episodes), desc="Training", postfix={"mean_reward": 0})
        mean_eval_reward = 0
        for e in pbar:
            # Play one episode
            total_reward = self.play_one_episode()

            # Decay epsilon to play more greedily
            self.decay_epsilon()

            # Append to the external list total reward
            self.train_rewards.append(total_reward)
            mean_train_reward = np.mean(self.train_rewards[-100:])

            # Evaluate every some period
            if e % self.evaluation_period == 0:
                eval_rewards = []
                for _ in range(self.evaluation_episodes):
                    total_reward = self.play_one_episode(epsilon=1e-3)
                    eval_rewards.append(total_reward)
                mean_eval_reward = np.mean(eval_rewards)
                self.val_rewards.append(mean_eval_reward)
            pbar.set_postfix_str(
                f"Epsilon: {self.epsilon:.3f} | Train: {mean_train_reward:.2f} | Eval: {mean_eval_reward:.2f}"
            )

            # Check for early termination
            if mean_eval_reward > self.solved_threshold:
                pbar.close()
                print(f"Terminated after {e+1} steps with mean reward {mean_eval_reward:.2f}")
                break
