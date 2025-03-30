import numpy as np

class ExplorationStrategy:
    @staticmethod
    def softmax(q_values, tau=1.0):
        """Softmax function for action selection."""
        exp_values = np.exp((q_values - np.max(q_values)) / tau)
        return exp_values / np.sum(exp_values)
    
    @staticmethod
    def epsilon_greedy(q_values, epsilon):
        """
        Epsilon-greedy action selection.
        """
        if np.random.rand() < epsilon:
            return np.random.choice(len(q_values))  # Random action
        return np.argmax(q_values)  # Greedy action
    
    @staticmethod
    def linear_decay(start, end, episode, total_episodes):
        """Linear decay function."""
        return start - (start - end) * (episode / total_episodes)
    
    @staticmethod
    def exponential_decay(start, end, episode, decay_rate):
        """Exponential decay function."""
        return end + (start - end) * np.exp(-decay_rate * episode)