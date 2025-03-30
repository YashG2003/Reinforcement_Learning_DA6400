import numpy as np
from utils.exploration import ExplorationStrategy

class SARSAAgent:
    def __init__(self, state_space_shape, action_space_n):
        self.q_table = self._initialize_q_table(state_space_shape + (action_space_n,))
        self.exploration = ExplorationStrategy()
    
    @staticmethod
    def _initialize_q_table(shape):
        """Initialize the Q-table with zeros."""
        return np.zeros(shape)
    
    def get_action(self, state, epsilon):
        """Select action using epsilon-greedy policy."""
        return self.exploration.epsilon_greedy(self.q_table[state], epsilon)
    
    def update(self, state, action, reward, next_state, next_action, gamma, learning_rate):
        """SARSA update rule."""
        td_target = reward + gamma * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += learning_rate * td_error