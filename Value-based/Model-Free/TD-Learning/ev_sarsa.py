import numpy as np
from tdlearning import TDLearning

class EVSARSA(TDLearning):
    def __init__(self, alpha: float, gamma: float, epsilon: float, action_space):
        super().__init__(alpha, gamma, epsilon, action_space)

    def get_ev(self, state):
        '''
        Get the action probabilities for the given state.
        This is a epsilon-greedy policy.
        The action with the highest Q-value is selected with probability 1 - epsilon.
        The other actions are selected with probability epsilon / (number of actions).
        '''
        q_values = np.array([self.getQ(state, action) for action in self.action_space])
        max_idx = np.argmax(q_values)
        action_probs = np.ones(len(self.action_space)) * self.epsilon / len(self.action_space)
        action_probs[max_idx] += 1 - self.epsilon
        return np.sum(q_values.dot(action_probs))
    
    def update(self, state, action, reward, nextState):
        '''
        SARSA update rule:
            TD Target = reward + \gamma * V(s')
        '''
        current_q = self.getQ(state, action)
        next_V = self.get_ev(nextState)
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_V)
        self.setQ(state, action, new_q)