import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame # Added for rendering

class ConnectFourEnv(gym.Env):
    """
    Gymnasium environment for the game of Connect Four.

    Args:
        rows (int): Number of rows in the board (default: 6).
        cols (int): Number of columns in the board (default: 7).
        win_length (int): Number of pieces in a row needed to win (default: 4).
        render_mode (str, optional): The rendering mode ('human' or 'rgb_array'). Defaults to None.

    Attributes:
        observation_space (gym.spaces.Box): The observation space is the board state.
        action_space (gym.spaces.Discrete): The action space is the column index.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, rows=6, cols=7, win_length=4, opponnent: str = "smart", render_mode=None):
        super().__init__()

        if rows < win_length or cols < win_length:
             raise ValueError(f"Board dimensions ({rows}x{cols}) must be at least "
                              f"win_length ({win_length}) in size.")

        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        # Player 1 is the agent (usually learns)
        self.agent_player = 1
        # Player -1 is the opponent (rule-based or other)
        self.opponent_player = -1
        # Agent starts
        self.current_player = self.agent_player
        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        # Observation: Board state (0: empty, 1: player 1, -1: player 2)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.rows, self.cols),
                                            dtype=np.int8)

        # Action: Choose a column (0 to cols-1)
        self.action_space = spaces.Discrete(self.cols)
        # Rewards
        self.reward_win = 1
        self.reward_draw = 0
        self.reward_lose = -1
        self.reward_step = 0

        # --- Pygame Rendering Attributes ---
        self.window = None
        self.clock = None
        self.cell_size = 100 # Size of each cell in pixels for rendering
        self.colors = {
            0: (200, 200, 200), # Empty cells (light gray)
            1: (255, 0, 0),    # Player 1 (Red)
            -1: (255, 255, 0)  # Player 2 (Yellow)
        }
        self.board_color = (0, 0, 255) # Board structure (Blue)
        self.screen_width = self.cols * self.cell_size
        self.screen_height = (self.rows + 1) * self.cell_size # Extra row for dropping animation/selection
        self.strat = opponnent

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        
        if self.render_mode == "human":
            self._render_frame()

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        """Takes a step in the environment for the AGENT."""
        if self.current_player != self.agent_player:
             raise Exception("It's not the agent's turn!")

        if not self.action_space.contains(action):
             raise ValueError(f"Invalid action: {action}. Action must be in {self.action_space}")

        # --- Agent's Move ---
        if not self._is_valid_action(action):
            # Agent chose an invalid column - penalize heavily and end game? Or just return penalty?
            # Let's penalize and end the game for simplicity here.
            terminated = True
            reward = self.reward_lose # Or a specific large negative reward
            observation = self._get_obs()
            info = self._get_info()
            info["error"] = "Invalid move: Column full."
            # print(f"Agent {self.agent_player} made invalid move {action}") # Debug
            return observation, reward, terminated, False, info # Terminated=True, Truncated=False

        # Place agent's piece
        self._place_piece(self.agent_player, action)

        # Check if agent won
        if self._check_win(self.agent_player):
            observation = self._get_obs()
            reward = self.reward_win
            terminated = True
            info = self._get_info()
            return observation, reward, terminated, False, info

        # Check for draw
        if self._is_board_full():
            observation = self._get_obs()
            reward = self.reward_draw
            terminated = True
            info = self._get_info()
            return observation, reward, terminated, False, info

        # Switch to opponent's turn
        self.current_player = self.opponent_player

        # --- Opponent's Move ---
        opponent_action = self._opponent_move()

        # Place opponent's piece (we assume the strategy returns a valid move)
        self._place_piece(self.opponent_player, opponent_action)

        # Check if opponent won
        if self._check_win(self.opponent_player):
            observation = self._get_obs()
            reward = self.reward_lose # Agent lost
            terminated = True
            info = self._get_info()
            return observation, reward, terminated, False, info

        # Check for draw again (opponent might fill the board)
        if self._is_board_full():
            observation = self._get_obs()
            reward = self.reward_draw
            terminated = True
            info = self._get_info()
            return observation, reward, terminated, False, info

        # Switch back to agent's turn for the next step
        self.current_player = self.agent_player

        # Game continues, return standard step reward
        observation = self._get_obs()
        reward = self.reward_step # Often 0, agent gets reward only at the end
        terminated = False
        info = self._get_info() # Get info for the *agent's* next turn
        return observation, reward, terminated, False, info


    def _find_drop_row(self, column):
        """Finds the lowest empty row in a given column. Returns -1 if full."""
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, column] == 0:
                return r
        return -1

    def _place_piece(self, player, column):
        """Places a piece for the given player in the given column."""
        row = self._find_drop_row(column)
        if row == -1:
             # This should ideally not happen if _is_valid_action is checked first,
             # especially for the agent. The opponent strategy should also ensure validity.
             raise ValueError(f"Internal Error: Attempted to place piece in full column {column}.")
        self.board[row, column] = player
        if self.render_mode == "human":
            self._render_frame()


    def _opponent_move(self):
        """Returns the column chosen by the opponent."""
        valid_actions = [col for col in range(self.cols) if self._is_valid_action(col)]
        if self.strat == "random":
            return self.np_random.choice(valid_actions)
        # 1. Check for immediate win for the opponent
        for action in valid_actions:
            row = self._find_drop_row(action)
            self.board[row, action] = self.opponent_player # Simulate move
            if self._check_win(self.opponent_player):
                self.board[row, action] = 0 # Undo simulation
                return action # Take winning move
            self.board[row, action] = 0 # Undo simulation
        # 2. Check for lose on the next step
        for action in valid_actions:
            row = self._find_drop_row(action)
            self.board[row, action] = self.agent_player # Simulate as if the agent moves 
            if self._check_win(self.agent_player):
                self.board[row,action] = 0
                return action
            self.board[row, action] = 0
        # 3. Random move
        return self.np_random.choice(valid_actions)

    def _is_valid_action(self, column):
        """Checks if the top row of the column is empty."""
        return self.board[0, column] == 0

    def _get_valid_actions(self):
        """Returns a mask of valid actions (1 for valid, 0 for invalid)."""
        return np.array([1 if self._is_valid_action(col) else 0 for col in range(self.cols)], dtype=np.int8)

    def _check_win(self, player):
        """Checks if the given player has won."""
        # Check horizontal
        for r in range(self.rows):
            for c in range(self.cols - self.win_length + 1):
                if np.all(self.board[r, c:c + self.win_length] == player):
                    return True

        # Check vertical
        for r in range(self.rows - self.win_length + 1):
            for c in range(self.cols):
                if np.all(self.board[r:r + self.win_length, c] == player):
                    return True

        # Check positive diagonal (\)
        for r in range(self.rows - self.win_length + 1):
            for c in range(self.cols - self.win_length + 1):
                if np.all(np.diag(self.board[r:r + self.win_length, c:c + self.win_length]) == player):
                    return True

        # Check negative diagonal (/)
        for r in range(self.win_length - 1, self.rows):
            for c in range(self.cols - self.win_length + 1):
                 # Create a subgrid and check its anti-diagonal
                 subgrid = self.board[r - self.win_length + 1 : r + 1, c : c + self.win_length]
                 # Use np.fliplr to flip left-right, then np.diag to get the anti-diagonal
                 if np.all(np.diag(np.fliplr(subgrid)) == player):
                     return True

        return False

    def _is_board_full(self):
        """Checks if the board is completely full."""
        return not np.any(self.board == 0)

    def _get_obs(self):
        """Gets the current observation (board state)."""
        board = self.board.copy() # Return a copy
        return board.flatten()
        
    def _get_info(self):
        """Gets auxiliary information, including the action mask for the current player."""
        return {"action_mask": self._get_valid_actions()}

    def render(self):
        """Renders the environment based on the selected render_mode."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        # Human mode rendering is handled within step/reset

    def _render_frame(self):
        """Renders one frame of the environment."""
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="human")'
            )
            return

        try:
            import pygame
        except ImportError as e:
            raise gym.error.DependencyNotInstalled(
                 "pygame is not installed, run `pip install gymnasium[classic_control]`"
            ) from e


        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Connect Four")
            else:  # rgb_array
                self.window = pygame.Surface((self.screen_width, self.screen_height))
            if self.clock is None:
                self.clock = pygame.time.Clock()


        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill(self.board_color) # Fill background with board color

        # Draw the pieces
        for r in range(self.rows):
            for c in range(self.cols):
                # Calculate center of the circle
                center_x = int(c * self.cell_size + self.cell_size / 2)
                center_y = int(r * self.cell_size + self.cell_size / 2 + self.cell_size) # Offset down by one cell size
                radius = int(self.cell_size / 2 * 0.85) # Leave some padding

                # Draw circle for the piece based on board state
                pygame.draw.circle(
                    canvas,
                    self.colors[self.board[r, c]],
                    (center_x, center_y),
                    radius,
                )
                # Draw a black border around the circle slot (optional, looks nicer)
                pygame.draw.circle(
                    canvas,
                    (0,0,0), # Black border
                    (center_x, center_y),
                    radius,
                    width=2 # Border width
                )


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump() # Process other events now
            pygame.display.update() # Actually update the screen

            # We need to ensure that human-rendering occurs at the defined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
             # Transpose needed because pygame coords are (x,y) but numpy is (row, col) -> (y,x)
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """Closes the rendering window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None



gym.register(
    id='ConnectFour-v0',                                # Unique ID for your env
    entry_point='connect4:ConnectFourEnv',              # Module path and class name
    # Optional: Specify default arguments for __init__
    # kwargs={'rows': 6, 'cols': 7, 'win_length': 4},
    # Optional: Set a maximum number of steps per episode
    max_episode_steps=100, # e.g., (rows * cols) for max possible moves
    # Optional: Set a reward threshold for considering the env solved
    # reward_threshold=0.95, # Example threshold (depends on reward scheme)
)