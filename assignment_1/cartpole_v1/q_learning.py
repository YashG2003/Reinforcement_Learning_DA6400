import numpy as np
from discretization import create_bins, discretize_state
import wandb

def initialize_q_table(shape):
    """
    Initialize the Q-table with zeros.
    shape: The dimensions of the Q-table (num_bins for each state variable x num_actions).
    """
    return np.zeros(shape)

def softmax(q_values, tau=1.0):
    """
    Softmax function for action selection.
    q_values: Q-values for the current state.
    tau: Controls exploration (higher tau = more exploration).
    Returns a probability distribution over actions.
    """
    # Subtract max for numerical stability
    exp_values = np.exp((q_values - np.max(q_values)) / tau)
    return exp_values / np.sum(exp_values)

def linear_decay(start, end, episode, total_episodes):
    """
    Linear decay function.
    start: Initial value.
    end: Final value.
    episode: Current episode.
    total_episodes: Total number of episodes.
    """
    return start - (start - end) * (episode / total_episodes)

def exponential_decay(start, end, episode, decay_rate):
    """
    Exponential decay function.
    start: Initial value.
    end: Final value.
    episode: Current episode.
    decay_rate: Rate of decay.
    """
    return end + (start - end) * np.exp(-decay_rate * episode)

def q_learning(env, bins, num_episodes=1000, learning_rate=0.1, gamma=0.99, tau_start=1.0, tau_end=0.01, decay_type='linear', decay_rate=0.01):
    """
    Q-Learning algorithm with Softmax exploration and tau decay.
    env: The Gym environment.
    bins: The bins for discretizing the state space.
    num_episodes: Number of episodes to run.
    learning_rate: Learning rate (alpha).
    gamma: Discount factor.
    tau_start: Initial value of tau.
    tau_end: Final value of tau.
    decay_type: Type of decay ('linear' or 'exponential').
    decay_rate: Rate of decay for exponential decay.
    Returns a list of scores for each episode.
    """
    # Initialize Q-table
    q_table = initialize_q_table((len(bins[0]), len(bins[1]), len(bins[2]), len(bins[3]), env.action_space.n))
    scores = []
    mean_scores = []
    max_mean_score = -np.inf

    for episode in range(num_episodes):
        # Decay tau
        if decay_type == 'linear':
            tau = linear_decay(tau_start, tau_end, episode, num_episodes)
        elif decay_type == 'exponential':
            tau = exponential_decay(tau_start, tau_end, episode, decay_rate)
        else:
            tau = tau_start  # No decay

        state, _ = env.reset()  # Extract the state and ignore the info dictionary
        state = discretize_state(state, bins)  # Discretize the state
        done = False
        total_score = 0

        while not done:
            # Softmax action selection
            action_probs = softmax(q_table[state], tau)
            action = np.random.choice(env.action_space.n, p=action_probs)

            # Take the action and observe the next state and reward
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state, bins)
            total_score += reward

            # Q-Learning update
            best_next_action = np.argmax(q_table[next_state])
            q_table[state + (action,)] += learning_rate * (reward + gamma * q_table[next_state + (best_next_action,)] - q_table[state + (action,)])

            # Move to the next state
            state = next_state

        # Log the total score for the episode
        scores.append(total_score)
        mean_score = np.mean(scores[-100:])  # Moving average over last 100 episodes
        mean_scores.append(mean_score)
        max_mean_score = max(max_mean_score, mean_score)

        # Log metrics
        wandb.log({
            "Episode": episode,
            "Score": total_score,
            "Mean_Score": mean_score,
            "Max_Mean_Score": max_mean_score
        })

    return scores