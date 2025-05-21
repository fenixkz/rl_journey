import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PER import PrioritizedReplayBuffer
from D3QN import D3QNCNN
import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import FireResetEnv
import ale_py

#  Hyperparameters
N_EPISODES = 50001
GAMMA = 0.99
ALPHA = 0.5
BATCH_SIZE = 128
BUFFER_SIZE = 100000
BETA_START = 0.5 # Starting beta for PER
BETA_END = 1 # Ending beta for PER
BETA_ANNEALING_STEPS = 100000 # Steps over which to anneal beta
LEARNING_RATE = 5e-4
TARGET_MODEL_UPDATE_PERIOD = 500 # Every N steps update target model
EPSILON_START = 1.0
EPSILON_END = 0.1
DECAY_PORTION = 0.5  # 60% of episodes
decay_steps = int(N_EPISODES * DECAY_PORTION)
EPSILON_DECAY = (EPSILON_END / EPSILON_START) ** (1 / decay_steps)
VALIDATE_PERIOD = 100 # Every N episodes validate model

resume_training = False

device = "cuda" if torch.cuda.is_available() else "cpu"

def learn(online_model: nn.Module, target_model: nn.Module, buffer: PrioritizedReplayBuffer, optimizer: optim.Optimizer, beta: float):
    # 1. Sample batch from PER buffer
    # Returns: state, action, reward, next_state, done, weights, indices
    state, action, reward, next_state, done, indices, weights = buffer.sample(BATCH_SIZE, beta)

    # 2. Convert to tensors and move to device
    state = torch.FloatTensor(np.float32(state)).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device) 
    action = torch.LongTensor(action).to(device) # Actions are indices, should be Long
    reward = torch.FloatTensor(reward).to(device)
    # done flags are often treated as floats (0.0 or 1.0) for masking
    done = torch.FloatTensor(done.astype(np.float32)).to(device)
    weights = torch.FloatTensor(weights).to(device) # Importance sampling weights

    # 3. Calculate Q(s, a) for the actions taken
    # Get Q-values for all actions from online model
    q_values_all_actions = online_model(state)
    # Use gather to select the Q-value for the specific action taken
    q_s_a = q_values_all_actions.gather(1, action.unsqueeze(-1)).squeeze(-1) # Shape: [batch_size]

    # 4. Calculate DDQN Target
    with torch.no_grad(): # Target calculations should not affect gradients
        # --- DDQN Action Selection ---
        # Use online model to find the best action indices in the next state
        online_next_q_values = online_model(next_state)
        next_action_indices = torch.argmax(online_next_q_values, dim=1) # Shape: [batch_size]

        # --- DDQN Action Evaluation ---
        # Use target model to get Q-values for all actions in the next state
        target_next_q_values_all = target_model(next_state)
        # Select the Q-value from the target network corresponding to the action chosen by the online network
        q_s_prime_a_prime = target_next_q_values_all.gather(1, next_action_indices.unsqueeze(-1)).squeeze(-1) # Shape: [batch_size]

        # --- Calculate TD Target ---
        # Target = r + gamma * Q_target(s', argmax_a' Q_online(s', a')) * (1 - done)
        # Multiply by (1 - done) so target is just 'r' if next_state is terminal
        td_target = reward + GAMMA * q_s_prime_a_prime * (1 - done)

    # 5. Calculate Loss 
    # Calculate TD Error: td_target - q_s_a
    td_error = td_target - q_s_a

    # Calculate loss: (TD Error)^2 weighted by IS weights
    # Using detach() on td_target ensures gradients only flow through q_s_a
    loss = (weights * (td_error.pow(2))).mean()

    # 6. Update Priorities in PER Buffer
    priorities = torch.abs(td_error.detach()) + 1e-6 # Add small epsilon for stability
    buffer.update_priorities(indices, priorities.cpu().numpy())

    # 7. Perform Gradient Descent Step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def get_beta(total_steps):
    """Calculates the current beta for PER annealing."""
    fraction = min(total_steps / BETA_ANNEALING_STEPS, 1.0)
    return BETA_START + fraction * (BETA_END - BETA_START)

def update_target(online_model: nn.Module, target_model: nn.Module):
    target_model.load_state_dict(online_model.state_dict())

def choose_action(online_model: nn.Module, epsilon: float, state: np.ndarray):
    if np.random.random() < epsilon:
        return np.random.choice(env.action_space.n)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
            q_values = online_model(state_tensor)
            # Get action index with max q-value
            action = torch.argmax(q_values, dim=1).squeeze().item() 
            return action

def validate(online_model: nn.Module, env: gym.Env):
    rewards = []
    for i in range(10):
        state = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            action = choose_action(online_model, 0.0, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)

# ENV_NAME = "ALE/Pong-v5"
ENV_NAME = "ALE/Breakout-v5"
name = ENV_NAME.split('/')[-1]

env = gym.make(ENV_NAME)
env = FireResetEnv(env)
env = gym.wrappers.AtariPreprocessing(
    env,
    noop_max=0,
    frame_skip=1,
    terminal_on_life_loss=True,
    screen_size=84,
    grayscale_obs=True
)
env = gym.wrappers.FrameStackObservation(env, stack_size=4)

print(f"Observation Space: {env.observation_space}")
print(f"Action Space: {env.action_space}")

online_model = D3QNCNN(env.observation_space.shape, env.action_space.n).to(device)
target_model  = D3QNCNN(env.observation_space.shape, env.action_space.n).to(device)

update_target(online_model, target_model) # Load weights from online model to target model
target_model.eval() # Set it to eval, we do not use any dropout or batch norm, but it is a good practice

    
optimizer = optim.Adam(online_model.parameters(), lr = LEARNING_RATE)

buffer = PrioritizedReplayBuffer(BUFFER_SIZE, ALPHA)

print("\n--- Starting Training ---")
print(f"Using {device}")


total_steps = 0
test_rewards = []
train_rewards = []
train_loss = []

if resume_training:
    checkpoint = torch.load(f"{name}_checkpoint.pth")
    online_model.load_state_dict(checkpoint["model_state_dict"])
    
    update_target(online_model, target_model) # Load weights from online model to target model
    epsilon = checkpoint['epsilon']
    episode_start = checkpoint['episode']
    best_test_reward = checkpoint['best_test_reward']
else:
    epsilon = EPSILON_START
    best_test_reward = -float('inf')
    episode_start = 0

# ---------- MAIN TRAIN LOOP ----------------
for e in range(episode_start, N_EPISODES):
    state = env.reset()[0]
    done = False
    episode_reward = 0
    episode_loss = 0
    learning_steps = 0
    while not done:
        total_steps += 1
        # Choose action given our epsilon greedy policy
        action = choose_action(online_model, epsilon, state)
        # Step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Record data into buffer
        buffer.push(state, action, reward, next_state, done)
        # Move to next state
        state = next_state
        episode_reward += reward

        # Update target model if time has come
        if total_steps % TARGET_MODEL_UPDATE_PERIOD == 0:
            update_target(online_model, target_model)

        # If enough samples, then backpropogate and learn
        if len(buffer) > BATCH_SIZE:
            # Calculate current beta
            current_beta = get_beta(total_steps)
            loss = learn(online_model, target_model, buffer, optimizer, current_beta)
            learning_steps += 1
            episode_loss += loss
    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    avg_loss = episode_loss / learning_steps if learning_steps > 0 else 0
    print(f"Episode {e}/{N_EPISODES}, Episode Reward: {episode_reward}, Episode Mean loss: {avg_loss}, Current Epsilon: {epsilon}")
    if e % VALIDATE_PERIOD == 0:
        with torch.no_grad():
            test_reward = validate(online_model, env)
            print(f"Mean validation reward: {test_reward}")
            test_rewards.append(test_reward)
        if test_reward > best_test_reward:
            # Save the online model after training
            model_save_path = f"{name}_checkpoint.pth"
            checkpoint = {
                'model_state_dict': online_model.state_dict(),
                'epsilon': epsilon,
                'episode': e,
                'best_test_reward': best_test_reward
            }
            torch.save(checkpoint, model_save_path)
            
            print(f"Best model saved to {model_save_path}")
            best_test_reward = test_reward
    train_rewards.append(episode_reward)
    train_loss.append(avg_loss)


import matplotlib.pyplot as plt
# --- Plotting ---
window_size = 50 # Use a larger window for smoother average
# Define moving average function if not defined elsewhere
def moving_avg(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if len(train_rewards) >= window_size:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(window_size - 1, len(train_rewards)), moving_avg(train_rewards, window_size), label=f"Train {window_size}-ep moving avg reward")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Episode Rewards (Moving Average)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(10 - 1, len(test_rewards)), moving_avg(test_rewards, 10), label=f"Test {window_size}-ep moving avg reward")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Episode Rewards (Moving Average)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)

    loss_window = 50 # Smooth loss over more steps
    if len(train_loss) >= loss_window:
        plt.plot(range(loss_window - 1, len(train_loss)), moving_avg(train_loss, loss_window), label=f"{loss_window}-step moving avg loss", alpha=0.7)
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss (Moving Average)")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("Not enough episodes to plot moving average.")

env.close()

