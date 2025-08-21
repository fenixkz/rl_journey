import os
import random
import sys
from abc import abstractmethod
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from core.configs import AgentConfig
from core.memory import NStepPrioritizedReplayBuffer, NStepReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer
from core.policies import (
    CNNNetwork,
    DuelingCNN,
    DuelingFC,
    FCNetwork,
)

current_path = os.path.dirname(__file__)
parent_path = os.path.join(current_path, "../../../")
sys.path.append(os.path.abspath(parent_path))

from common.base_agent import BaseAgent  # noqa: E402
from common.utils.atari_utils import auto_fire, clip_reward, get_atari_env  # noqa: E402


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DQNBase(BaseAgent):

    def __init__(
        self,
        env_id: gym.Env,
        agent_config: AgentConfig,
        is_atari: bool,
        solved_threshold: float,
    ):
        if is_atari:
            env = get_atari_env(env_id)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "Detected non-discrete action space, this class works only with discrete action space problems!"

        super().__init__(env=env, solved_threshold=solved_threshold)

        # --------- HYPERPARAMS ---------
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.gamma = agent_config.gamma
        self.epsilon = agent_config.max_epsilon
        self.min_epsilon = agent_config.min_epsilon
        self.epsilon_decay = agent_config.eps_decay
        self.batch_size = agent_config.batch_size
        self.lr = agent_config.lr
        self.hard_update = agent_config.hard_target_update
        self.target_update_freq = agent_config.target_update_freq
        self.tau = agent_config.tau
        self.learning_starts = agent_config.learning_starts
        self.learning_freq = agent_config.learning_freq
        self.is_atari = is_atari
        self.evaluation_period = agent_config.evaluation_period
        self.evaluation_episodes = agent_config.evaluation_episodes

        self.name = "N/A"
        # Set seed for reproduction of results
        self.seed = agent_config.seed
        # Params for N-step return
        self.n_step_return = agent_config.n_step_return
        self.n_step_buffer = deque(maxlen=self.n_step_return)
        # ---------

        # --------- POLICY ---------
        if not is_atari:
            print(
                f"Detected a classic (vector) environment, \
                observation space shape: {self.observation_space.shape}, \
                using Fully Connected (FC) Network"
            )
            if agent_config.dueling:
                self.online_model = DuelingFC(
                    self.observation_space.shape[0], self.action_space.n, agent_config.hidden_dim
                ).to(self.device)
                self.target_model = DuelingFC(
                    self.observation_space.shape[0], self.action_space.n, agent_config.hidden_dim
                ).to(self.device)
            else:
                self.online_model = FCNetwork(
                    self.observation_space.shape[0], self.action_space.n, agent_config.hidden_dim
                ).to(self.device)
                self.target_model = FCNetwork(
                    self.observation_space.shape[0], self.action_space.n, agent_config.hidden_dim
                ).to(self.device)
        else:
            print(
                f"Detected an Atari environment, \
                observation space shape: {self.observation_space.shape}, \
                using Convolutional Neural Network (CNN)"
            )
            if agent_config.dueling:
                self.online_model = DuelingCNN(
                    self.observation_space.shape, self.action_space.n, agent_config.hidden_dim
                ).to(self.device)
                self.target_model = DuelingCNN(
                    self.observation_space.shape, self.action_space.n, agent_config.hidden_dim
                ).to(self.device)
            else:
                self.online_model = CNNNetwork(
                    self.observation_space.shape, self.action_space.n, agent_config.hidden_dim
                ).to(self.device)
                self.target_model = CNNNetwork(
                    self.observation_space.shape, self.action_space.n, agent_config.hidden_dim
                ).to(self.device)

        # Copy the initial weights at the beginning
        self.hard_update_target_network()
        # ---------

        # Initialize a replay buffer
        if agent_config.memory == "rb":
            if agent_config.n_step_return == 1:
                self.memory = ReplayBuffer(size=agent_config.memory_size)
            else:
                self.memory = NStepReplayBuffer(size=agent_config.memory_size)
        elif agent_config.memory == "per":
            if agent_config.n_step_return == 1:
                self.memory = PrioritizedReplayBuffer(size=agent_config.memory_size, alpha=agent_config.alpha)
            else:
                self.memory = NStepPrioritizedReplayBuffer(size=agent_config.memory_size, alpha=agent_config.alpha)
        # Initialize optimizer, use same set of params as in Nature paper
        self.optimizer = torch.optim.RMSprop(
            self.online_model.parameters(), lr=self.lr, alpha=0.95, eps=0.01, momentum=0.95, centered=False
        )

    def choose_action(self, state: np.ndarray, epsilon: float = None):
        """
        Epsilon-greedy policy of action choosing.
        """
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.random() < epsilon:
            return np.random.choice(self.action_space.n)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_model(state_tensor)
            action = torch.argmax(q_values, dim=1).squeeze().item()
            return action

    def decay_epsilon(self):
        """
        Nature paper decay epsilon linear instead of exponentially
        """
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.min_epsilon

    def clip_reward(self, reward):
        """Implements the DeepMind reward clipping. Turn all reward either to +1 or -1"""
        return clip_reward(reward)

    def auto_fire(self):
        return auto_fire(self.env)

    def save(self, save_path: str):
        """
        Save online model for later evaluation
        """
        os.makedirs(save_path, exist_ok=True)
        online_model_path = os.path.join(save_path, "online_model.pth")
        torch.save(self.online_model.state_dict(), online_model_path)

    def load(self, load_path: str):
        """
        Load pre-trained online model
        """
        os.makedirs(load_path, exist_ok=True)
        online_model_path = os.path.join(load_path, "online_model.pth")
        self.online_model.load_state_dict(torch.load(online_model_path))

    def soft_update_target_network(self):
        """
        Soft target network update
        """
        target_net_state_dict = self.target_model.state_dict()
        online_net_state_dict = self.online_model.state_dict()
        for key in online_net_state_dict:
            target_net_state_dict[key] = online_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                1 - self.tau
            )
        self.target_model.load_state_dict(target_net_state_dict)

    def hard_update_target_network(self):
        """
        Hard target network update
        """
        self.target_model.load_state_dict(self.online_model.state_dict())

    def update_target_network(self):
        if self.hard_update:
            return self.hard_update_target_network()
        return self.soft_update_target_network()

    def should_update_target(self, step: int):
        # If we are using hard update, then we must check whether it is time to update.
        if self.hard_update:
            return step % self.target_update_freq == 0
        # If we are using soft update, then we should do it every step
        return True

    def get_name(self):
        return self.name

    def _get_n_step_info(self):
        """Calculates the n-step return for the oldest transition in the buffer. Used for N-step-return"""
        reward, next_state, done = 0, None, False

        # Sum discounted rewards
        for i in range(len(self.n_step_buffer)):
            # Get a reward for a single instance in the deque
            r = self.n_step_buffer[i][2]
            # Accumulate discounted rewards
            reward += (self.gamma**i) * r

            # If we hit a terminal state before N steps were executed, the return is calculated up to that point
            if self.n_step_buffer[i][4]:  # done flag
                # Final next state is then the next_state of this specific experience tuple (transition)
                next_state = self.n_step_buffer[i][3]
                # Done is true
                done = True
                # The effective number of steps is i + 1 (because loop starts from 0 and not from 1)
                return reward, next_state, done, i + 1

        # If no terminal state was hit, the next_state is from the last transition
        next_state = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]
        return reward, next_state, done, len(self.n_step_buffer)

    def evaluate(self):
        """
        Play a complete episode in the environment using the current policy.
        """
        rewards = []
        for _ in range(self.evaluation_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state=state, epsilon=0.0)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward
            rewards.append(total_reward)
        return np.mean(rewards)

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def train(self, max_steps: int = 100000, timeout: float = None):
        pass
