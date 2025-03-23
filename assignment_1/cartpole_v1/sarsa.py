import numpy as np
from discretization import create_bins, discretize_state
import wandb

def initialize_q_table(shape):
    """
    Initialize the Q-table with zeros.
    shape: The dimensions of the Q-table (num_bins for each state variable x num_actions).
    """
    return np.zeros(shape)

def epsilon_greedy(q_values, epsilon):
    """
    Epsilon-greedy action selection.
    q_values: Q-values for the current state.
    epsilon: Exploration rate.
    Returns the selected action.
    """
    if np.random.rand() < epsilon:
        return np.random.choice(len(q_values))  # Explore: random action
    else:
        return np.argmax(q_values)  # Exploit: best action

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

def sarsa(env, bins, num_episodes=1000, learning_rate=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, decay_type='linear', decay_rate=0.01):
    """
    SARSA algorithm with epsilon-greedy exploration and epsilon decay.
    env: The Gym environment.
    bins: The bins for discretizing the state space.
    num_episodes: Number of episodes to run.
    learning_rate: Learning rate (alpha).
    gamma: Discount factor.
    epsilon_start: Initial value of epsilon.
    epsilon_end: Final value of epsilon.
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
        # Decay epsilon
        if decay_type == 'linear':
            epsilon = linear_decay(epsilon_start, epsilon_end, episode, num_episodes)
        elif decay_type == 'exponential':
            epsilon = exponential_decay(epsilon_start, epsilon_end, episode, decay_rate)
        else:
            epsilon = epsilon_start  # No decay

        state, _ = env.reset()  # Extract the state and ignore the info dictionary
        state = discretize_state(state, bins)  # Discretize the state
        done = False
        total_score = 0

        # Epsilon-greedy action selection
        action = epsilon_greedy(q_table[state], epsilon)

        while not done:
            # Take the action and observe the next state and reward
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state, bins)
            total_score += reward

            # Epsilon-greedy action selection for the next state
            next_action = epsilon_greedy(q_table[next_state], epsilon)

            # SARSA update
            q_table[state + (action,)] += learning_rate * (reward + gamma * q_table[next_state + (next_action,)] - q_table[state + (action,)])

            # Move to the next state and action
            state = next_state
            action = next_action

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