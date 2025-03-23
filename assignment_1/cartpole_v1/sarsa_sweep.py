import gymnasium
import wandb
from sarsa import sarsa
from discretization import create_bins

def train_sarsa():
    """
    Train the SARSA algorithm using WandB for hyperparameter tuning.
    """
    wandb.init()
    config = wandb.config

    # Generate a descriptive name for the run
    run_name = f"nb_{config.num_bins}_lr_{config.learning_rate}_epsilon_end_{config.epsilon_end}_decay_type_{config.decay_type}"
    wandb.run.name = run_name

    # Create the environment and bins
    env = gymnasium.make("CartPole-v1")
    bins = create_bins(num_bins=config.num_bins)

    # Run SARSA
    rewards = sarsa(
        env, bins,
        num_episodes=config.num_episodes,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        decay_type=config.decay_type,
        decay_rate=config.decay_rate
    )

    # Close the environment
    env.close()

# SARSA Sweep Configuration
sarsa_sweep_configuration = {
    'method': 'bayes',
    'name': 'sarsa_sweep_1',
    'metric': {'goal': 'maximize', 'name': 'Max_Mean_Score'},
    'parameters': {
        'num_bins': {'values': [10, 20, 30]},
        'num_episodes': {'values': [10000]},
        'learning_rate': {'values': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]},
        'gamma': {'values': [0.99]},
        'epsilon_start': {'values': [1.0]},
        'epsilon_end': {'values': [1e-1, 1e-2]},
        'decay_type': {'values': ['linear', 'exponential']},
        'decay_rate': {'values': [0.01, 0.1, 1.0]}
    }
}

# Run SARSA Sweep
sarsa_sweep_id = wandb.sweep(sweep=sarsa_sweep_configuration, project="rl_a1")
wandb.agent(sarsa_sweep_id, function=train_sarsa, count=100)