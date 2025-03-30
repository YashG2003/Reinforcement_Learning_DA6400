import gymnasium as gym
import wandb
from trainers.sarsa_trainer import SARSATrainer

def train():
    wandb.init()
    config = wandb.config
    
    # Generate descriptive run name
    run_name = (
        f"nb_{config.num_bins}_lr_{config.learning_rate}_"
        f"eps_end_{config.epsilon_end}_decay_{config.decay_type}"
    )
    wandb.run.name = run_name
    
    # Create environment and run training
    env = gym.make("CartPole-v1")
    trainer = SARSATrainer(env, dict(config))
    avg_metrics = trainer.train()
    
    # Log episode-wise metrics
    for episode in range(config.num_episodes):
        wandb.log({
            "Episodic_Reward": avg_metrics['avg_scores'][episode],
            "Mean_Episodic_Reward": avg_metrics['avg_mean_scores'][episode],
            "Regret": avg_metrics['avg_regrets'][episode],
            "Cumulative_Regret": avg_metrics['avg_cumulative_regrets'][episode]
        })
        
    wandb.log({
        "Final_Cumulative_Regret": avg_metrics['Final_Cumulative_Regret']
    })
    
    env.close()

# SARSA sweep configuration
sweep_config = {
    'method': 'bayes',
    'name': 'cartpole_sarsa',
    'metric': {'goal': 'minimize', 'name': 'Final_Cumulative_Regret'},
    'parameters': {
        'num_bins': {'values': [20, 30]},
        'num_episodes': {'values': [10000]},
        'num_runs': {'values': [5]},
        'learning_rate': {'values': [0.1, 0.5, 0.01]},
        'gamma': {'values': [0.99]},
        'epsilon_start': {'values': [1.0]},
        'epsilon_end': {'values': [0.1, 0.5, 0.01]},
        'decay_type': {'values': ['linear', 'exponential']},
        'decay_rate': {'values': [0.01, 0.1]}
    }
}

# Run sweep
sweep_id = wandb.sweep(sweep_config, project="rl_a1", entity="da6400_rl")
wandb.agent(sweep_id, train, count=30)