import numpy as np

class StateDiscretizer:
    def __init__(self, num_bins=10):
        self.bins = self.create_bins(num_bins)
    
    @staticmethod
    def create_bins(num_bins=10):
        """Create bins for discretizing the continuous state space."""
        return [
            np.linspace(-4.8, 4.8, num_bins),  # Cart Position
            np.linspace(-4, 4, num_bins),      # Cart Velocity
            np.linspace(-0.418, 0.418, num_bins),  # Pole Angle
            np.linspace(-4, 4, num_bins)       # Pole Velocity
        ]
    
    def discretize(self, state):
        """Discretize a continuous state into discrete bins."""
        return tuple(
            np.digitize(state[i], self.bins[i]) - 1
            for i in range(len(state))
        )