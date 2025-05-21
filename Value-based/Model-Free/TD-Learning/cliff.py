import numpy as np
import time
import matplotlib.pyplot as plt

class CliffWorld():
    def __init__(self, height: int = 4, width: int = 12):
        # Map dimensions
        self.height = height
        self.width = width

        # Start position (bottom-left)
        self.pos_row = self.height - 1  # Y (row)
        self.pos_col = 0                # X (col)
        
        # Goal position (bottom-right)
        self.goal_row = self.height - 1
        self.goal_col = self.width - 1
        
        self.actions = np.array(["left", "right", "up", "down"])
        self.stateCount = self.height * self.width
        self.actionCount = len(self.actions)
        
        # Cliff positions: bottom row, columns 1 to width-2
        self.cliff_pos = np.array([(self.height - 1, i) for i in range(1, self.width - 1)])
    
    def reset(self):
        self.pos_row = self.height - 1
        self.pos_col = 0
        self.done = False
        return (self.pos_row, self.pos_col)
    
    def get_state_grid(self):
        """
        Returns a 2D numpy array representing the current state of the environment:
        0: empty, 1: cliff, 2: agent, 3: start, 4: goal
        """
        grid = np.zeros((self.height, self.width), dtype=int)
        # Mark cliff positions
        for (row, col) in self.cliff_pos:
            grid[row, col] = 1
        # Mark start
        grid[self.height - 1, 0] = 3
        # Mark goal
        grid[self.height - 1, self.width - 1] = 4
        # Mark agent
        grid[self.pos_row, self.pos_col] = 2
        return grid
        
    def step(self, action):
        # Move agent
        if action == "left":
            self.pos_col = self.pos_col - 1 if self.pos_col > 0 else self.pos_col
        elif action == "right":
            self.pos_col = self.pos_col + 1 if self.pos_col < self.width - 1 else self.pos_col
        elif action == "up":
            self.pos_row = self.pos_row - 1 if self.pos_row > 0 else self.pos_row
        elif action == "down":
            self.pos_row = self.pos_row + 1 if self.pos_row < self.height - 1 else self.pos_row

        # Check for cliff
        cliff = (self.pos_row, self.pos_col) in [tuple(pos) for pos in self.cliff_pos]
        win = (self.pos_row == self.goal_row and self.pos_col == self.goal_col)

        if cliff:
            reward = -10
            done = True
        elif win:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        return (self.pos_row, self.pos_col), reward, done
    