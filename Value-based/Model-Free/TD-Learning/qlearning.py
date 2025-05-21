import numpy as np
from tdlearning import TDLearning

class QLearningAgent(TDLearning):

    def __init__(self, alpha: float, gamma: float, epsilon: float, action_space):
        super().__init__(alpha, gamma, epsilon, action_space)
        
    def update(self, state, action, reward, nextState):
        '''
        Q-learning update rule, compute TD target as (r_t + max{Q(s_{t+1}, a)})
        '''
        current_q = self.getQ(state, action)
        next_max = max([self.getQ(nextState, a) for a in self.action_space])
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_max)
        self.setQ(state, action, new_q)
