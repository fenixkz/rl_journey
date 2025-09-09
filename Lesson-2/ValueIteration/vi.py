import gymnasium as gym
import numpy as np
from ttt import TicTacToeEnv

# Define the env
env = gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=True)
env = TicTacToeEnv()

# Get number of states (8x8=64)
n_states = env.observation_space.n
# Get number of possible actions (4)
n_actions = env.action_space.n
# Unwrap the env to get transition probabilities
base_env = env.unwrapped
# Get a matrix of transition probabilities
transition_prob = base_env.P
# Initialize a dictionary that hold a state-value for each state
# V*(s) for all s in S, initialized as 0 in the beginning
state_values = {s: 0 for s in range(n_states)}
# Discount factor
gamma = 0.9


def get_Q(state, action):
    """
    A function to compute the Q*(s,a)
    """
    Q = 0
    # Per formula in theory.md
    for prob, next_state, reward, _ in transition_prob[state][action]:
        Q += prob * (reward + gamma * state_values[next_state])
    return Q


def compute_V(state):
    """
    A function to compute V*(s)
    """
    # Per formula in theory.md
    return max(get_Q(state, action) for action in range(n_actions))


def choose_action(state):
    """
    A function to choose the action that maximizes the Q*(s,a)
    """
    # Chooses the best action that corresponds to the highest Q value
    return max(range(n_actions), key=lambda action: get_Q(state, action))


# Total number of iterations to perform
num_iter = 100
for i in range(num_iter):
    # Re-calculate state-values
    new_state_values = {s: compute_V(s) for s in range(n_states)}
    # Get a maximum difference between old estimates and new
    diff = max(abs(new_state_values[s] - state_values[s]) for s in range(n_states))
    print(f"Iteration {i}, diff: {diff}")
    state_values = new_state_values
    # If the difference is low, it means we are close to convergence
    if diff < 0.001:
        break

n_episodes = 100
n_steps = 200
# Play the episode with our estimates of values
rewards = []
for i in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0
    for j in range(n_steps):
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        total_reward += reward
        if terminated or truncated:
            break
    rewards.append(total_reward)

print(f"Average reward per {n_episodes} is {np.mean(rewards)}")

# --- 2. PLAY AGAINST THE TRAINED AGENT ---

play_again = "y"
while play_again.lower() == "y":
    state, _ = env.reset()
    terminated = False
    print("\n--- New Game: You are 'O', the AI is 'X' ---")
    env.render()

    while not terminated:
        # --- AI's Turn ---
        current_state = env._board_to_state(env.board)
        ai_action = choose_action(current_state)

        # Apply AI's action to the board
        row, col = divmod(ai_action, 3)
        if env.board[row, col] != 0:
            print("Error: AI tried to play on an occupied cell. This shouldn't happen.")
            break
        env.board[row, col] = env.agent_player

        print(f"AI plays at position {ai_action}:")
        env.render()

        # Check for AI win or draw
        if env._check_winner(env.board, env.agent_player):
            print("AI wins!")
            terminated = True
            continue
        if env._is_draw(env.board):
            print("It's a draw!")
            terminated = True
            continue

        # --- Human's Turn ---
        human_action = -1
        while human_action == -1:
            try:
                move = int(input("Enter your move (0-8): "))
                if 0 <= move <= 8:
                    row, col = divmod(move, 3)
                    if env.board[row, col] == 0:
                        human_action = move
                    else:
                        print("That cell is already occupied. Try again.")
                else:
                    print("Invalid input. Please enter a number between 0 and 8.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Apply human's action
        row, col = divmod(human_action, 3)
        env.board[row, col] = env.opponent_player
        env.render()

        # Check for human win or draw
        if env._check_winner(env.board, env.opponent_player):
            print("Congratulations, you win!")
            terminated = True
        if env._is_draw(env.board) and not terminated:
            print("It's a draw!")
            terminated = True

    play_again = input("Play again? (y/n): ")

print("Thanks for playing!")
