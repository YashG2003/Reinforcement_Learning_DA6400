import gymnasium
import wandb
from q_learning import q_learning
from discretization import create_bins

def train_q_learning():
    """
    Train the Q-Learning algorithm using WandB for hyperparameter tuning.
    """
    wandb.init()
    config = wandb.config

    # Generate a descriptive name for the run
    run_name = f"nb_{config.num_bins}_lr_{config.learning_rate}_tau_start_{config.tau_start}_decay_type_{config.decay_type}"
    wandb.run.name = run_name

    # Create the environment and bins
    env = gymnasium.make("CartPole-v1")
    bins = create_bins(num_bins=config.num_bins)

    # Run Q-Learning
    rewards = q_learning(
        env, bins,
        num_episodes=config.num_episodes,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        tau_start=config.tau_start,
        tau_end=config.tau_end,
        decay_type=config.decay_type,
        decay_rate=config.decay_rate
    )

    # Close the environment
    env.close()

# Q-Learning Sweep Configuration
q_learning_sweep_configuration = {
    'method': 'bayes',
    'name': 'q_learning_sweep_2',
    'metric': {'goal': 'maximize', 'name': 'Max_Mean_Score'},
    'parameters': {
        'num_bins': {'values': [10, 20, 30]},
        'num_episodes': {'values': [10000]},
        'learning_rate': {'values': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]},
        'gamma': {'values': [0.99]},
        'tau_start': {'values': [1, 1e1, 1e2, 1e3, 1e4, 1e5]},
        'tau_end': {'values': [0.01]},
        'decay_type': {'values': ['linear', 'exponential']},
        'decay_rate': {'values': [0.01, 0.1, 1.0]}
    }
}

# Run Q-Learning Sweep
q_learning_sweep_id = wandb.sweep(sweep=q_learning_sweep_configuration, project="rl_a1")
wandb.agent(q_learning_sweep_id, function=train_q_learning, count=100)