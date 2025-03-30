import numpy as np
from utils.exploration import ExplorationStrategy

class QLearningAgent:
    def __init__(self, state_space_shape, action_space_n):
        self.q_table = self._initialize_q_table(state_space_shape + (action_space_n,))
        self.exploration = ExplorationStrategy()
    
    @staticmethod
    def _initialize_q_table(shape):
        """Initialize the Q-table with zeros."""
        return np.zeros(shape)
    
    def get_action(self, state, tau=1.0):
        """Select action using softmax policy."""
        action_probs = self.exploration.softmax(self.q_table[state], tau)
        return np.random.choice(len(action_probs), p=action_probs)
    
    def update(self, state, action, reward, next_state, gamma, learning_rate):
        """Q-learning update rule."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += learning_rate * td_error