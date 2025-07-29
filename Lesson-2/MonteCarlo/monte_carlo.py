import numpy as np


class MCLearning:

    def __init__(self, alpha: float, gamma: float, epsilon: float, action_space):
        self.alpha = alpha 
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space = action_space
        # Our Q-values are stored in a dictionary, that's why we can apply it to a small set of problems with low state space
        self._Q = {}

    def getQ(self, state, action):
        '''
        A function to get the Q-values of provided (state, action). If not present in the dictionary,
        then default value is zero.
        '''
        return self._Q.get((state, action), 0.0)
    
    def setQ(self, state, action, q):
        '''
        Set the Q-value for the provided (state, action) pair
        '''
        self._Q[(state, action)] = q

    def get_action(self, state):
        '''
        Use epsilon-greedy policy to choose an action
        '''
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            q_values = [self.getQ(state, action) for action in self.action_space]
            return self.action_space[np.argmax(q_values)]
    
    def learn(self, state, action, g):
        '''
        Function to update our estimation of Q-values which is basically a return.
        Use a moving average to smoothly update the estimate
        '''
        self.setQ(state, action, self.getQ(state, action) + self.alpha * (g - self.getQ(state, action)))