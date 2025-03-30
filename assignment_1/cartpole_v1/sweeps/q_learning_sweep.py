import gymnasium as gym
import numpy as np
import wandb
from trainers.q_learning_trainer import QLearningTrainer

def train():
    wandb.init()
    config = wandb.config
    
    # Generate descriptive run name
    run_name = (
        f"nb_{config.num_bins}_lr_{config.learning_rate}_"
        f"tau_start_{config.tau_start}_decay_type_{config.decay_type}"
    )
    wandb.run.name = run_name
    
    # Create environment and run training
    env = gym.make("CartPole-v1")
    trainer = QLearningTrainer(env, dict(config))
    avg_metrics = trainer.train()
    
    # Still log individual metrics for table view
    for episode in range(config.num_episodes):
        wandb.log({
            "Episodic_Reward": avg_metrics['avg_scores'][episode],
            "Mean_Episodic_Reward": avg_metrics['avg_mean_scores'][episode],
            "Regret": avg_metrics['avg_regrets'][episode],
            "Cumulative_Regret": avg_metrics['avg_cumulative_regrets'][episode],
        })
    
    wandb.log({
        "Final_Cumulative_Regret": avg_metrics['Final_Cumulative_Regret']
    })
    
    env.close()

# Sweep configuration remains the same
sweep_config = {
    'method': 'bayes',
    'name': 'cartpole_q_learning_h1_1',
    'metric': {'goal': 'minimize', 'name': 'Final_Cumulative_Regret'},
    'parameters': {
        'num_bins': {'values': [30]},
        'num_episodes': {'values': [10000]},
        'num_runs': {'values': [5]},
        'learning_rate': {'values': [1e-1]},
        'gamma': {'values': [0.99]},
        'tau_start': {'values': [1e1]},
        'tau_end': {'values': [0.01]},
        'decay_type': {'values': ['exponential']},
        'decay_rate': {'values': [1e-3]}
    }
}

# Run sweep
sweep_id = wandb.sweep(sweep_config, project="rl_a1", entity="da6400_rl")
wandb.agent(sweep_id, train, count=1)