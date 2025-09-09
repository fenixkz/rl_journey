import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tqdm import tqdm


class TicTacToeEnv(gym.Env):
    """
    A simple Tic-Tac-Toe environment that conforms to the Gymnasium API.
    The agent plays as 'X' (represented by 1) and the opponent plays as 'O'
    (represented by 2). The opponent follows a random policy.

    This version pre-computes the transition probability matrix 'P' upon
    initialization to be compatible with value and policy iteration algorithms.

    State representation:
    The 3x3 board is flattened into a 9-element array. Each element can be:
    - 0: Empty
    - 1: Agent's mark ('X')
    - 2: Opponent's mark ('O')
    This array is then converted to a single integer to represent the state,
    treating the board as a base-3 number. This gives 3^9 = 19,683 unique states.

    Action space:
    The action space is Discrete(9), where each action corresponds to placing a
    mark in one of the 9 cells of the board (0-8).

    Rewards:
    - +10 for winning the game
    - -10 for losing the game
    -  -1 for placing a mark on an already occupied cell (invalid move)
    -   0 for a draw
    -   0 for an ongoing game
    """

    metadata = {"render.modes": ["console"]}

    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        # The board is a 3x3 grid
        self.board = np.zeros((3, 3), dtype=int)
        # Agent is player 1 ('X'), opponent is player 2 ('O')
        self.agent_player = 1
        self.opponent_player = 2

        # Define action and observation space
        # Actions are 0-8, corresponding to the 9 cells
        self.action_space = spaces.Discrete(9)
        # Observation space is the total number of possible board configurations (3^9)
        self.observation_space = spaces.Discrete(3**9)

        # Pre-compute the transition probability matrix
        print("Computing transition probabilities, this may take a moment...")
        self.P = self._compute_transition_prob()
        print("Done.")

    def _board_to_state(self, board):
        """Converts the 3x3 board to a unique integer state."""
        state = 0
        for i, cell in enumerate(board.flatten()):
            state += cell * (3**i)
        return int(state)

    def _state_to_board(self, state):
        """Converts a unique integer state back to a 3x3 board."""
        flat_board = np.zeros(9, dtype=int)
        temp_state = state
        for i in range(8, -1, -1):
            power_of_3 = 3**i
            cell_value = temp_state // power_of_3
            flat_board[i] = cell_value
            temp_state %= power_of_3
        return flat_board.reshape((3, 3))

    def _check_winner(self, board, player):
        """Check if the specified player has won on the given board."""
        # Check rows
        for row in range(3):
            if all(board[row, :] == player):
                return True
        # Check columns
        for col in range(3):
            if all(board[:, col] == player):
                return True
        # Check diagonals
        if all(np.diag(board) == player) or all(np.diag(np.fliplr(board)) == player):
            return True
        return False

    def _is_draw(self, board):
        """Check if the game is a draw on the given board."""
        return not np.any(board == 0)

    def _compute_transition_prob(self):
        """
        Computes the transition probability matrix P.
        P[state][action] = list of (prob, next_state, reward, terminated)
        """
        n_states = self.observation_space.n
        n_actions = self.action_space.n
        P = {s: {a: [] for a in range(n_actions)} for s in range(n_states)}

        for state in tqdm(range(n_states), desc="Calculating Transitions"):
            for action in range(n_actions):
                current_board = self._state_to_board(state)
                row, col = divmod(action, 3)

                # Case 1: Agent makes an invalid move (cell is not empty)
                if current_board[row, col] != 0:
                    P[state][action].append((1.0, state, -1, True))
                    continue

                # --- Simulate Agent's Move ---
                board_after_agent = current_board.copy()
                board_after_agent[row, col] = self.agent_player

                # Case 2: Agent wins with this move
                if self._check_winner(board_after_agent, self.agent_player):
                    next_state = self._board_to_state(board_after_agent)
                    P[state][action].append((1.0, next_state, 10, True))
                    continue

                # Case 3: Draw after agent's move
                if self._is_draw(board_after_agent):
                    next_state = self._board_to_state(board_after_agent)
                    P[state][action].append((1.0, next_state, 0, True))
                    continue

                # --- Simulate Opponent's Random Move ---
                available_opponent_moves = np.where(board_after_agent.flatten() == 0)[0]
                prob = 1.0 / len(available_opponent_moves)

                for opp_action in available_opponent_moves:
                    opp_row, opp_col = divmod(opp_action, 3)
                    board_after_opponent = board_after_agent.copy()
                    board_after_opponent[opp_row, opp_col] = self.opponent_player

                    # Case 4a: Opponent wins
                    if self._check_winner(board_after_opponent, self.opponent_player):
                        next_state = self._board_to_state(board_after_opponent)
                        P[state][action].append((prob, next_state, -10, True))
                    # Case 4b: Draw after opponent's move
                    elif self._is_draw(board_after_opponent):
                        next_state = self._board_to_state(board_after_opponent)
                        P[state][action].append((prob, next_state, 0, True))
                    # Case 4c: Game continues
                    else:
                        next_state = self._board_to_state(board_after_opponent)
                        P[state][action].append((prob, next_state, 0, False))
        return P

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        self.board = np.zeros((3, 3), dtype=int)
        observation = self._board_to_state(self.board)
        info = {}
        return observation, info

    def step(self, action):
        """Execute one time step within the environment."""
        # The step function now uses the pre-computed P matrix for its logic.
        # This is not how it's typically used in live play, but it demonstrates
        # consistency with the P matrix.
        # A random outcome is chosen based on the probabilities.

        transitions = self.P[self._board_to_state(self.board)][action]

        probs = [t[0] for t in transitions]
        # Choose one of the possible outcomes based on the probabilities
        outcome_idx = np.random.choice(len(transitions), p=probs)
        prob, next_state, reward, terminated = transitions[outcome_idx]

        self.board = self._state_to_board(next_state)
        truncated = False  # This environment does not truncate
        info = {}

        return next_state, reward, terminated, truncated, info

    def render(self, mode="console"):
        """Renders the environment."""
        if mode != "console":
            raise NotImplementedError()

        # Map numbers to symbols for display
        symbols = {0: ".", 1: "X", 2: "O"}

        # Print the board
        for row in range(3):
            print(" ".join([symbols[cell] for cell in self.board[row]]))
        print("-" * 5)
