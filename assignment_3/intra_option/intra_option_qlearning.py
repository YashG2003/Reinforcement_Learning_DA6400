import numpy as np

class IntraOptionQAgent:
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

    def update(self, state, action, next_state, reward, consistent_options, terminates):
        """
        Update Q-values for all options that would have selected the same action
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            consistent_options: List of options that would have selected this action
            terminates: Dictionary mapping options to whether they terminate in next_state
        """
        for option in consistent_options:
            # If option terminates at next_state, bootstrap from max Q-value
            if terminates[option]:
                target = reward + self.gamma * np.max(self.Q[next_state])
            # Otherwise, bootstrap from the same option's value
            else:
                target = reward + self.gamma * self.Q[next_state, option]
                
            # Update Q-value
            self.Q[state, option] += self.alpha * (target - self.Q[state, option])


