import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 0.1
        self.epsilon_decay=0.9999
        self.epsilon_min=0.005

    def select_action(self, state, eps):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        '''
        if np.random.random_sample() > self.eps:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(self.nA)
        '''
        self.eps = eps
        probs = np.ones(self.nA) * self.eps / self.nA
        probs[np.argmax(self.Q[state])] = 1 - self.eps + (self.eps / self.nA)
        return np.random.choice(self.nA, p=probs)
        
    def updateQsarsamax(self, alpha, gamma, state, action, reward, next_state=None):
        Q_s_a = self.Q[state][action]
        Q_s_a_next = np.max(self.Q[next_state]) if next_state is not None else 0
        return Q_s_a + alpha * (reward + gamma * Q_s_a_next - Q_s_a)

    def updateQexpsarsa(self, alpha, gamma, state, action, reward, next_state=None):
        Q_s_a = self.Q[state][action]
        policy_s = np.ones(self.nA) * self.eps / self.nA
        policy_s[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA)
        Q_s_a_next = np.dot(self.Q[next_state], policy_s)
        return Q_s_a + alpha * (reward + gamma * Q_s_a_next - Q_s_a)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        alpha = 0.11
        gamma = 0.9
        self.Q[state][action] = self.updateQsarsamax(alpha, gamma, state, action, reward, next_state)
#         self.Q[state][action] = self.updateQexpsarsa(alpha, gamma, state, action, reward, next_state)