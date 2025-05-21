import numpy as np
from abc import ABC, abstractmethod

class TDLearning(ABC):
    def __init__(self, alpha: float, gamma: float, epsilon: float, action_space):
        self.alpha = alpha 
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space = action_space
        self._Q = {}

    def getQ(self, state, action):
        return self._Q.get((state, action), 0.0)
    
    def setQ(self, state, action, q):
        self._Q[(state, action)] = q

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            q_values = [self.getQ(state, action) for action in self.action_space]
            return self.action_space[np.argmax(q_values)]
    
    @abstractmethod
    def update(self, state, action, reward, nextState):
        pass