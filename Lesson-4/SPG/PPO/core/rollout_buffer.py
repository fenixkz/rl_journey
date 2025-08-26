import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, device, gamma, gae_lambda):
        self.device = device
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Instead of moving those from cpu to gpu every add(), keep them on cpu
        self.states = torch.zeros((num_steps, num_envs) + obs_shape)
        self.rewards = torch.zeros((num_steps, num_envs))
        self.dones = torch.zeros((num_steps, num_envs))

        # Those are computed by the policy, so they are already of correct device
        self.actions = torch.zeros((num_steps, num_envs), device=self.device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=self.device)
        self.values = torch.zeros((num_steps, num_envs), device=self.device)

        # GAE results can also be calculated on GPU
        self.advantages = torch.zeros((num_steps, num_envs), device=self.device)
        self.returns = torch.zeros((num_steps, num_envs), device=self.device)

    def add(
        self,
        step: int,
        states: np.ndarray,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: torch.Tensor,
    ):
        # Cast np array to tensors
        self.states[step] = torch.tensor(states)
        self.rewards[step] = torch.tensor(rewards)
        self.dones[step] = torch.tensor(dones)
        # Those are tensors already
        self.actions[step] = actions
        self.logprobs[step] = logprobs
        self.values[step] = values

    def reset(self):
        # Reset back from .gpu() to .cpu()
        self.states = self.states.to("cpu")
        self.rewards = self.rewards.to("cpu")
        self.dones = self.dones.to("cpu")

    @torch.no_grad()
    def compute_returns_and_advantages(self, next_value: torch.Tensor):
        last_gae_lambda = torch.zeros(self.num_envs).to(self.device)

        # To avoid mix device error
        self.rewards = self.rewards.to(self.device)
        self.dones = self.dones.to(self.device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_vals = next_value
            else:
                next_vals = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_vals * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae_lambda = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
            )

        self.returns = self.advantages + self.values

    def get_mini_batches(self, num_mini_batches: int):
        batch_size = self.num_steps * self.num_envs
        indices = np.arange(batch_size)
        np.random.shuffle(indices)
        mini_batch_size = batch_size // num_mini_batches

        # Flatten all data
        # Note: .flatten(0, 1) or .reshape(batch_size, ...) works here
        flat_states = self.states.flatten(0, 1).to(self.device)
        flat_actions = self.actions.reshape(batch_size)
        flat_logprobs = self.logprobs.reshape(batch_size)
        flat_advantages = self.advantages.reshape(batch_size)
        flat_returns = self.returns.reshape(batch_size)
        flat_values = self.values.reshape(batch_size)

        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mini_batch_indices = indices[start:end]

            # Transfer to GPU only when the mini-batch is needed
            yield {
                "states": flat_states[mini_batch_indices],
                "actions": flat_actions[mini_batch_indices],
                "logprobs": flat_logprobs[mini_batch_indices],
                "advantages": flat_advantages[mini_batch_indices],
                "returns": flat_returns[mini_batch_indices],
                "values": flat_values[mini_batch_indices],
            }
