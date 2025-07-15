import torch
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from agent import PPOAgent
import gymnasium as gym
from collections import deque
import torch.optim as optim
def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

class PPOTrainer:
    """
    PPO trainer with vector environments
    """
    
    def __init__(self, 
                env_name='CartPole-v1',
                num_envs=4,
                num_steps=2, 
                critic_loss_coef = 0.5,
                actor_lr=1e-3,
                critic_lr=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                entropy_coef=0.01,
                hidden_size=64,
                max_steps=200000, 
                num_mini_batches=4,
                clip_epsilon = 0.2,
                update_epochs = 5,
                seed=42,
                normalize_advantage=False,
                anneal_lr=False,
                clip_values=False,
                solved_threshold=495.0, 
                ):
        # Hyperparameters
        self.env_name = env_name
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.max_steps = int(max_steps)
        self.solved_threshold = solved_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.normalize_advantage = normalize_advantage
        self.anneal_lr = anneal_lr
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = num_envs * num_steps
        self.mini_batch_size = int(self.batch_size // num_mini_batches)
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.clip_values = clip_values
        self.critic_loss_coef = critic_loss_coef

        # Set seed
        set_seed(seed)
        
        # Gym env details
        self.envs = gym.vector.SyncVectorEnv([make_env(env_name) for _ in range(num_envs)])
        obs_dim = self.envs.single_observation_space.shape[0]
        action_dim = self.envs.single_action_space.n
        # PPO agent and optimizer
        self.agent = PPOAgent(obs_dim, action_dim, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.agent.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.agent.critic.parameters(), lr=critic_lr, eps=1e-5)
        # Pre-allocated buffers for data that does NOT need a computation graph
        self.obs = torch.zeros((num_steps, num_envs, obs_dim)).to(self.device)
        self.actions = torch.zeros((num_steps, num_envs) + self.envs.single_action_space.shape, dtype=torch.long).to(self.device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(self.device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(self.device)
        self.dones = torch.zeros((num_steps, num_envs)).to(self.device)
        self.values = torch.zeros((num_steps, num_envs)).to(self.device)
        self.current_states = None # To keep track of current states in transition between rollouts

        # Performance metrics
        self.current_episode_rewards = np.zeros(self.num_envs, dtype=np.float64)
        self.env_rewards = {i: [] for i in range(self.num_envs)}
        self.episode_rewards = []
        self.global_steps = 0

    def train(self):
        num_iterations = self.max_steps // self.batch_size
        # Start the game
        states, _ = self.envs.reset(seed=self.seed)
        self.current_states = states 

        # Loop over total number of steps
        for step in range(1, num_iterations):
            # Anneal the learning rate
            if self.anneal_lr:
                frac = 1.0 - (step - 1.0) / self.max_steps
                lrnow = frac * self.actor_lr
                self.actor_optimizer.param_groups[0]["lr"] = lrnow
                lrnow = frac * self.critic_lr
                self.critic_optimizer.param_groups[0]["lr"] = lrnow

            # Collect rollouts and get back the tensors with computation graphs
            self.collect_rollouts()

            # Get a value for the next state after rollout has stepped M times, for GAE calculations
            with torch.no_grad():
                next_values = self.agent.evaluate(self.current_states).squeeze(-1) # The critic returns (num_envs, 1), so we need to squeeze last dim
            
            # Calculate returns and advantages using GAE
            returns, advantages = self.compute_returns_and_advantages(next_values)

            # Update the agent
            self.learn(returns, advantages)
            # exit()

            if np.mean(self.episode_rewards[-20:]) > self.solved_threshold:
                print("Env has been solved!")
                break

        return self.episode_rewards
    
    def collect_rollouts(self):
        """
        Collects `num_steps` of experience from each environment.
        """
        # Get the state at which the previous rollout has ended
        states = self.current_states
        for i in range(self.num_steps):
            # Track the total number of steps done globally
            self.global_steps += self.num_envs
            
            # Sample actions, get log_probs, entropy, and state-values. Do not track gradients, we are only doing inference
            with torch.no_grad():
                actions, log_probs, _, values, logits, probs = self.agent.act_and_evaluate(states)
                # print("Debugging session")
                # print(f"actions: {actions}")
                # print(f"log_probs: {log_probs}")
                # print(f"values: {values}")
                # print(f"logits: {logits}")
                # print(f"probs: {probs}")
                
            # Step in the environments
            next_states, rewards, terminated, truncated, _ = self.envs.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated)
            
            # Store data
            self.actions[i] = actions # It is tensor
            # print(f"Step {i} actions: {actions}")
            self.logprobs[i] = log_probs # It is tensor
            # print(f"Step {i} log_probs: {log_probs}")
            self.values[i] = values.squeeze(-1) # The critic returns (num_envs, 1), so we need to squeeze last dim
            # print(f"Step {i} values: {self.values[i]}")
            self.obs[i] = torch.FloatTensor(states).to(self.device) # states are np.array (returned by env)
            # print(f"Step {i} obs: {self.obs[i]}")
            self.rewards[i] = torch.tensor(rewards, dtype=torch.float32).to(self.device).view(-1) # Also assure that it contains only 1 dim
            # print(f"Step {i} rewards: {self.rewards[i]}")
            self.dones[i] = torch.tensor(dones, dtype=torch.float32).to(self.device).view(-1) # Also assure that it contains only 1 dim
            # print(f"Step {i} dones: {self.dones[i]}")
            # Transit in the environment
            states = next_states 
            # Traking the performance
            self.current_episode_rewards += rewards
            for env_idx, done in enumerate(dones):
                if done:
                    # An episode finished in this environment.
                    # 1. Log the completed episode's total reward.
                    final_reward = self.current_episode_rewards[env_idx]
                    self.env_rewards[env_idx].append(final_reward)
                    self.episode_rewards.append(final_reward)
                    if env_idx == 0:
                        print(f"[ENV {env_idx}] Episode: {len(self.env_rewards[env_idx])}, Episode Total Reward: {final_reward}. Mean over last 20 episodes over all envs: {np.mean(self.episode_rewards[-20:])}")
                    # 2. Reset the trackers for this specific environment.
                    self.current_episode_rewards[env_idx] = 0
        # Store the last states after the rollout has finished, such that next rollout knows where to start
        self.current_states = states
        
    def compute_returns_and_advantages(self, next_values):
        """
        Compute returns and advantages using GAE from collected rollout data.
        Args:
            next_values: Values for the states after the last rollout step.
        Returns:
            returns: N-step returns (targets for the critic).
            advantages: GAE advantages (weights for the actor).
        """
        # Initialize advantages, shape (M,N)
        advantages = torch.zeros_like(self.values).to(self.device)
        # print(f"Initialized advantages: {advantages}")
        # Iterate in reverse order
        last_gae_lambda = 0
        # print(f"Starting GAE computation with last_gae_lambda: {last_gae_lambda}")
        for t in reversed(range(self.num_steps)):
            # print(f"Step {t}:")
            # The last element should have next_values computed explicitly 
            if t == self.num_steps - 1:
                next_vals = next_values
                # print(f"  Using next_values for next_vals: {next_vals}")
            else: # Or get the stored values of next index
                next_vals = self.values[t + 1]
                # print(f"  Using self.values[{t + 1}] for next_vals: {next_vals}")
            # If the done is True, then it means that the next state is terminal
            next_non_terminal = 1.0 - self.dones[t] 
            # print(f"  next_non_terminal: {next_non_terminal}")
            # Calculate deltas (td error)
            delta = self.rewards[t] + self.gamma * next_vals * next_non_terminal - self.values[t]
            # print(f"  delta: {delta}")
            # Calculate advantage using GAE
            advantages[t] = last_gae_lambda = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
            # print(f"  advantages[{t}]: {advantages[t]}")
            # print(f"  last_gae_lambda: {last_gae_lambda}")
        # Total return is Q = A + V
        returns = advantages + self.values
        # print(f"Final returns: {returns}")
        # print(f"Final advantages: {advantages}")
        return returns, advantages
    
    def learn(self, returns: torch.Tensor, advantages: torch.Tensor):
        # First flatten all collected data, from (M,N,...) to (M*N,...)
        batch_obs = self.obs.reshape((self.num_envs*self.num_steps, -1))
        batch_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        batch_values = self.values.reshape(-1)
        batch_logprobs = self.logprobs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)

        # print(f"batch_obs: {batch_obs}")
        # print(f"batch_actions: {batch_actions}")
        # print(f"batch_values: {batch_values}")
        # print(f"batch_logprobs: {batch_logprobs}")
        # print(f"batch_advantages: {batch_advantages}")
        # print(f"batch_returns: {batch_returns}")

        # Construct batch indices
        batch_indices = np.arange(self.batch_size)
        # print(f"batch_indices: {batch_indices}")
        for epoch in range(self.update_epochs):
            # print(f"Epoch: {epoch}")
            # Randomly shuffle batch indices
            np.random.shuffle(batch_indices)
            # print(f"Shuffled batch_indices: {batch_indices}")
            # Now do mini-batch gradient descent
            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_indices = batch_indices[start:end]
                # print(f"Mini-batch indices: {mini_batch_indices}")
                # Do forward pass and get the logprobs of actions that we took, entropy and state-values, track gradients because we are going to do backpropogation
                _, new_log_probs, new_entropy, new_values, logits, probs = self.agent.act_and_evaluate(batch_obs[mini_batch_indices], batch_actions[mini_batch_indices].long())
                # print(f"new_log_probs: {new_log_probs}")
                # print(f"new_entropy: {new_entropy}")
                # print(f"new_values: {new_values}")
                # print(f"new logits: {logits}")
                # print(f"new probs: {probs}")
                # Use logarithm property: r_t = pi_new / pi_old -> log(r_t) = log(pi_new) - log(pi_old)
                logratio = new_log_probs - batch_logprobs[mini_batch_indices]
                # print(f"logratio: {logratio}")
                # r_t = e^(log(r_t))
                ratio = logratio.exp()
                # print(f"ratio: {ratio}")
                # if start == 0:
                #     print(f"Ratio must be 1. Ratio: {ratio}")

                mini_batch_advantages = batch_advantages[mini_batch_indices]
                # print(f"mini_batch_advantages (before norm): {mini_batch_advantages}")
                if self.normalize_advantage: # Note: we do normalization on mini-batch level
                    mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages.mean()) / (mini_batch_advantages.std() + 1e-8) # To avoid division by zero
                    # print(f"mini_batch_advantages (after norm): {mini_batch_advantages}")
                
                # Actor (policy) loss
                unclamped_loss = -mini_batch_advantages * ratio
                clamped_loss = -mini_batch_advantages * torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
                # print(f"unclamped_loss: {unclamped_loss}")
                # print(f"clamped_loss: {clamped_loss}")
                actor_loss = torch.max(unclamped_loss, clamped_loss).mean() # Use max because of negative sign
                # print(f"actor_loss (before entropy): {actor_loss}")

                # Critic (value) loss
                new_values = new_values.squeeze(-1) # Remove last dim
                # print(f"new_values (squeezed): {new_values}")
                # Clip values
                if self.clip_values:
                    critic_loss_unclipped = (new_values - batch_returns[mini_batch_indices]) ** 2 # Standard MSE loss
                    value_clipped = batch_values[mini_batch_indices] + torch.clamp(new_values - batch_values[mini_batch_indices], -self.clip_epsilon, self.clip_epsilon)
                    critic_loss_clipped = (value_clipped - batch_returns[mini_batch_indices]) ** 2
                    critic_loss_max = torch.max(critic_loss_unclipped, critic_loss_clipped)
                    # print(f"critic_loss_unclipped: {critic_loss_unclipped}")
                    # print(f"value_clipped: {value_clipped}")
                    # print(f"critic_loss_clipped: {critic_loss_clipped}")
                    # print(f"critic_loss_max: {critic_loss_max}")
                    critic_loss = 0.5 * critic_loss_max.mean()
                else:
                    critic_loss = 0.5 * ((new_values - batch_returns[mini_batch_indices]) ** 2).mean()
                    # print(f"critic_loss (no clipping): {critic_loss}")
                
                # Actor's backprop
                entropy_loss = new_entropy.mean()
                # print(f"entropy_loss: {entropy_loss}")
                actor_loss = actor_loss - self.entropy_coef*entropy_loss
                # print(f"actor_loss (final): {actor_loss}")
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                # Critic's backprop
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 0.5)
                self.critic_optimizer.step()