import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from core.configs import AgentConfig
from core.DQNBase import DQNBase
from tqdm import tqdm


class AgentDQN(DQNBase):
    """
    Double-DQN algorithm.
    """

    def __init__(self, env: gym.Env, agent_config: AgentConfig, solved_threshold: float, is_atari: bool):

        # Overwrite some specific params
        agent_config.memory = "rb"
        agent_config.dueling = False
        agent_config.n_step_return = 1

        # Initialize the parent class
        super().__init__(env, agent_config, is_atari, solved_threshold)
        # Overwrite the name
        self.name = "DDQN"

    def learn(self):
        # 1. Sample a batch of experience from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 2. Cast np.arrays into torch.Tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 3. Calculate Q(s,a) for all states in the batch and all possible actions. Shape: [batch_size, action_space.n]
        q_values: torch.Tensor = self.online_model(states)

        # 4. Use gather to select the Q-value for the specific actions in the batch
        actual_q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)  # Shape: [batch_size, 1]

        # 5. Calculate TD-target
        with torch.no_grad():  # Target calculations should not affect gradients, this is our target
            # --- DDQN Update Rule ---

            # 1. Use online model to find the indexes of the best action in the next states
            online_next_q = self.online_model(next_states)  # Shape: [batch_size, action_space.n]
            next_best_actions = torch.argmax(online_next_q, dim=1)  # Shape: [batch_size]

            # 2. Evaluate Q-values for the next states using target network
            # and pick Q-values according to the indexes of next_best_action
            target_new_q: torch.Tensor = self.target_model(next_states)  # Shape: [batch_size, action_space.n]
            target_q = target_new_q.gather(1, next_best_actions.unsqueeze(-1)).squeeze(-1)  # Shape: [batch_size]

            # 2. Calculate TD target
            # Target = reward + gamma * Q_target(s', argmax(Q_online(s'))) * (1 - done)
            # Multiply by (1 - done) so target is just 'r' if next_state is terminal
            td_target = rewards + self.gamma * target_q * (1 - dones)

        # 6. Calculate loss using Huber Loss (Smooth L1 Loss)
        # This correctly implements the TD-error clipping from the Nature paper.
        loss = F.mse_loss(actual_q_values, td_target)

        # 7. Perform Gradient Descent Step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def train(self, max_steps: int = 100000, timeout: float = None):
        obs, _ = self.env.reset()
        if self.is_atari:
            obs = self.auto_fire()

        pbar = tqdm(range(max_steps), desc="Training", postfix={"episde": 0, "mean_reward": "N/A", "avg_loss": "N/A"})

        episode = 0
        learning_steps = 0
        start_time = time.time()
        val_mean_reward = -float("inf")
        train_mean_reward = -float("inf")
        for global_step in pbar:
            action = self.choose_action(obs)
            self.decay_epsilon()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Clip rewards for Atari envs, per Nature paper
            clipped_reward = self.clip_reward(reward) if self.is_atari else reward
            # Push experience into memory
            self.memory.push(obs, action, clipped_reward, next_obs, terminated)

            if global_step > self.learning_starts and global_step % self.learning_freq == 0:
                self.learn()
                learning_steps += 1
                if self.should_update_target(learning_steps):
                    self.update_target_network()

            if done:
                if "episode" in info:
                    self.train_rewards.append(info["episode"]["r"])
                episode += 1
                train_mean_reward = np.mean(self.train_rewards[-100:])

                # Evaluate
                if episode % self.evaluation_period == 0:
                    val_mean_reward = self.evaluate()
                    self.val_rewards.append(val_mean_reward)
                    if val_mean_reward > self.solved_threshold:
                        print(f"Solved! Mean reward: {val_mean_reward}")
                        break

                obs, _ = self.env.reset()
                if self.is_atari:
                    obs = self.auto_fire()
            else:
                obs = next_obs
            # Log out the metrics
            postfix = {
                "Ep.": episode,
                "Train": f"{train_mean_reward:.2f}",
                "Val": f"{val_mean_reward:.2f}",
                "Eps.": f"{self.epsilon:.3f}",
            }
            pbar.set_postfix(postfix)

            if timeout is not None and time.time() - start_time > timeout * 60:
                print("[bold red] Timeout has expired, finishing the training...")
                break
        pbar.close()
