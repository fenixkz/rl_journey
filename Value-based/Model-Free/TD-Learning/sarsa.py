import numpy as np
from tdlearning import TDLearning

class SARSA(TDLearning):
    def __init__(self, alpha: float, gamma: float, epsilon: float, action_space):
        super().__init__(alpha, gamma, epsilon, action_space)

    def update(self, state, action, reward, nextState, nextAction):
        '''
        SARSA update rule:
            TD Target = reward + \gamma * Q(s_{t+1}, a_{t+1})
        '''
        current_q = self.getQ(state, action)
        next_q = self.getQ(nextState, nextAction)
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_q)
        self.setQ(state, action, new_q)