import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class WindyBridgeEnv(gym.Env):
    """
    A custom Gymnasium environment to demonstrate the difference between
    Q-Learning and SARSA. The agent must cross a chasm.
    - A short, narrow bridge is the optimal path but is risky.
    - A long, winding path is safe but suboptimal.
    - A "wind" mechanic pushes the agent off the bridge if it takes
      any non-optimal (exploratory) action while on the bridge.

    Attributes:
        metadata (dict): Metadata for rendering.
        action_space (spaces.Discrete): The discrete action space (0:Up, 1:Right, 2:Down, 3:Left).
        observation_space (spaces.Box): The agent's (row, col) position.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()

        self.grid_size = (7, 12)  # 7 rows, 12 columns
        self.wind_prob = 0.9  # 90% chance of being blown off the bridge on a non-optimal move

        # Define key locations
        self.start_pos = np.array([3, 0])
        self.goal_pos = np.array([3, 11])
        
        # The bridge is a single row of tiles
        self.bridge_coords = [(3, i) for i in range(1, 11)]
        
        # The chasm is below the bridge
        self.chasm_coords = [(4, i) for i in range(1, 11)]

        # Define action space: 0:Up, 1:Right, 2:Down, 3:Left
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([-1, 0]),  # Up
            1: np.array([0, 1]),   # Right
            2: np.array([1, 0]),   # Down
            3: np.array([0, -1]),  # Left
        }

        # Define observation space: agent's (row, col) position
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.grid_size[0] - 1, self.grid_size[1] - 1]),
            dtype=np.int32
        )

        # Rendering setup
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        # Provides the agent's distance to the goal, can be useful
        return {"distance": np.linalg.norm(self._agent_location - self.goal_pos)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = self.start_pos.copy()
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        
        # --- Wind Mechanic ---
        is_on_bridge = tuple(self._agent_location) in self.bridge_coords
        is_moving_forward = action == 1 # 1 is 'Right'

        if is_on_bridge and not is_moving_forward:
            if self.np_random.random() < self.wind_prob:
                # Blown into the chasm!
                self._agent_location = self.chasm_coords[0] # Fall to the first chasm tile
            else:
                # Resists the wind, move normally
                self._agent_location = np.clip(
                    self._agent_location + direction,
                    [0, 0],
                    [self.grid_size[0] - 1, self.grid_size[1] - 1]
                )
        else:
            # Normal movement
            self._agent_location = np.clip(
                self._agent_location + direction,
                [0, 0],
                [self.grid_size[0] - 1, self.grid_size[1] - 1]
            )

        # --- Determine reward and termination ---
        terminated = False
        reward = -1  # Cost for each step to encourage efficiency

        if tuple(self._agent_location) in self.chasm_coords:
            reward = -100
            terminated = True
        elif np.array_equal(self._agent_location, self.goal_pos):
            reward = 100
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size * self.grid_size[0] // self.grid_size[1]))
            pygame.display.set_caption("Windy Bridge Environment")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size * self.grid_size[0] // self.grid_size[1]))
        canvas.fill((255, 255, 255)) # White background
        pix_square_size = self.window_size / self.grid_size[1]

        # Draw goal
        pygame.draw.rect(
            canvas,
            (0, 255, 0), # Green
            pygame.Rect(
                pix_square_size * self.goal_pos[1],
                pix_square_size * self.goal_pos[0],
                pix_square_size,
                pix_square_size,
            ),
        )
        
        # Draw bridge
        for pos in self.bridge_coords:
            pygame.draw.rect(
                canvas,
                (210, 180, 140), # Tan
                pygame.Rect(
                    pix_square_size * pos[1],
                    pix_square_size * pos[0],
                    pix_square_size,
                    pix_square_size,
                ),
            )

        # Draw chasm
        for pos in self.chasm_coords:
            pygame.draw.rect(
                canvas,
                (0, 0, 128), # Navy Blue
                pygame.Rect(
                    pix_square_size * pos[1],
                    pix_square_size * pos[0],
                    pix_square_size,
                    pix_square_size,
                ),
            )

        # Calculate the center of the agent's circle in pixel coordinates (x, y)
        # agent_location[1] is the column (x), agent_location[0] is the row (y)
        agent_center_x = (self._agent_location[1] + 0.5) * pix_square_size
        agent_center_y = (self._agent_location[0] + 0.5) * pix_square_size

        # Draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255), # Blue
            (agent_center_x, agent_center_y), # Pass coordinates as a tuple
            pix_square_size / 3,
        )

        # Draw grid lines
        for x in range(self.grid_size[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size * self.grid_size[0] // self.grid_size[1]),
                width=2,
            )
        for y in range(self.grid_size[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * y),
                (self.window_size, pix_square_size * y),
                width=2,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()