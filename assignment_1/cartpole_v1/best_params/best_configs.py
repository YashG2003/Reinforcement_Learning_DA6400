best_qlearning_configs = [
    {
        'num_bins': 30,
        'learning_rate': 0.1,
        'gamma': 0.99,
        'tau_start': 10,
        'tau_end': 0.01,
        'decay_type': 'exponential',
        'decay_rate': 0.01,
        'num_episodes': 10000,
        'num_runs': 5
    },
    {
        'num_bins': 30,
        'learning_rate': 0.1,
        'gamma': 0.99,
        'tau_start': 10,
        'tau_end': 0.01,
        'decay_type': 'linear',
        'decay_rate': 0.001,
        'num_episodes': 10000,
        'num_runs': 5
    },
    {
        'num_bins': 30,
        'learning_rate': 0.1,
        'gamma': 0.99,
        'tau_start': 100,
        'tau_end': 0.01,
        'decay_type': 'exponential',
        'decay_rate': 0.1,
        'num_episodes': 10000,
        'num_runs': 5
    }
]

best_sarsa_configs = [
    {
        'num_bins': 30,
        'learning_rate': 0.5,
        'gamma': 0.99,
        'epsilon_start': 1,
        'epsilon_end': 0.1,
        'decay_type': 'exponential',
        'decay_rate': 0.01,
        'num_episodes': 10000,
        'num_runs': 5
    },
    {
        'num_bins': 30,
        'learning_rate': 0.1,
        'gamma': 0.99,
        'epsilon_start': 1,
        'epsilon_end': 0.1,
        'decay_type': 'exponential',
        'decay_rate': 0.1,
        'num_episodes': 10000,
        'num_runs': 5
    },
    {
        'num_bins': 30,
        'learning_rate': 0.1,
        'gamma': 0.99,
        'epsilon_start': 1,
        'epsilon_end': 0.1,
        'decay_type': 'linear',
        'decay_rate': 0.1,
        'num_episodes': 10000,
        'num_runs': 5
    }
]