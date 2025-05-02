import numpy as np

class SMDPQAgent:
    def __init__(self, n_states, n_options, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.Q = np.zeros((n_states, n_options))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_options = n_options

    def select_option(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_options)
        else:
            return np.argmax(self.Q[state])

    def update(self, s, o, reward_sum, next_state, duration):
        max_next = np.max(self.Q[next_state])
        target = reward_sum + (self.gamma ** duration) * max_next
        self.Q[s, o] += self.alpha * (target - self.Q[s, o])


