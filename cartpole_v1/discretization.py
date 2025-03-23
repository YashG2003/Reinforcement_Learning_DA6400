# discretization.py
import numpy as np

def create_bins(num_bins=10):
    """
    Create bins for discretizing the continuous state space.
    Returns a list of bins for each state variable.
    """
    bins = [
        np.linspace(-4.8, 4.8, num_bins),  # Cart Position
        np.linspace(-4, 4, num_bins),      # Cart Velocity
        np.linspace(-0.418, 0.418, num_bins),  # Pole Angle
        np.linspace(-4, 4, num_bins)       # Pole Velocity
    ]
    return bins

def discretize_state(state, bins):
    """
    Discretize a continuous state into discrete bins.
    state: The continuous state (a NumPy array or list of 4 values).
    bins: The bins for each state variable.
    Returns a tuple representing the discretized state.
    """
    discretized_state = []
    for i in range(len(state)):  # Iterate over each state variable
        discretized_state.append(np.digitize(state[i], bins[i]) - 1)  # Discretize and append
    return tuple(discretized_state)  # Return as a tuple for use as a dictionary key