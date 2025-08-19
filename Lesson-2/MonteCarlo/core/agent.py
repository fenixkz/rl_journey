import os
import sys
from typing import List, Optional

import numpy as np
from tqdm import tqdm

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../")
sys.path.append(os.path.abspath(parent_path))

from common.BaseTD import TDLearningAgent  # noqa: E402


class MCAgent(TDLearningAgent):

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
        )

    def learn(self, history: List):
        """
        Function to update our estimation of Q-values which is basically a return.
        Use a moving average to smoothly update the estimate
        """
        # First-Visit MC Update:
        # Update only once per (state, action) visit
        visited_pairs = set()
        # Total return of the episode: G
        g = 0
        # Calculate the return from the state_history in reverse
        for state, action, reward in history[::-1]:
            g = reward + self.gamma * g
            if (state, action) not in visited_pairs:
                current_q = self.getQ(state, action)
                updated_q = current_q + self.alpha * (g - current_q)
                self.setQ(state, action, updated_q)
                visited_pairs.add((state, action))

    def play_one_episode(self, epsilon: Optional[float] = None):
        """
        Play a complete episode in the environment using the current policy.
        """
        # Get initial state, note no seed is needed as it was seeded in the base class
        state, _ = self.env.reset()
        # Initialize an empty list to store the history of (state, action, reward) tuples
        history = []
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

            history.append((state, action, reward))
            # Transit to the next observation
            # Note next_state is continuos (if is_continuos)
            # So, state will be also continuous
            state = next_state
            # Sum total reward
            total_reward += reward
        return history, total_reward

    def train(self, num_episodes: int):
        """ """

        pbar = tqdm(range(num_episodes), desc="Training", postfix={"mean_reward": 0})
        mean_eval_reward = 0
        for e in pbar:
            # Play one episode and get the recorded data
            history, total_reward = self.play_one_episode()

            # Decay epsilon to play more greedily
            self.decay_epsilon()

            # Append to the external list total reward
            self.train_rewards.append(total_reward)
            mean_train_reward = np.mean(self.train_rewards[-100:])

            # Learn from that episode
            self.learn(history)

            # Evaluate the agent with pure greedy approach
            if e % self.evaluation_period == 0:
                eval_rewards = []
                for _ in range(self.evaluation_episodes):
                    _, total_reward = self.play_one_episode(epsilon=0.0)
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
