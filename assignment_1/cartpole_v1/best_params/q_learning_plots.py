import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from trainers.q_learning_trainer import QLearningTrainer
from best_params.best_configs import best_qlearning_configs

def train_configurations():
    """Train all configurations and save results to pickle file"""
    env = gym.make("CartPole-v1")
    results = []
    
    try:
        for config in best_qlearning_configs:
            print(f"\nTraining with config: {config}")
            trainer = QLearningTrainer(env, config)
            avg_metrics = trainer.train()
            
            # Store only what we need for plotting
            results.append({
                'config': config,
                'episodes': np.arange(config['num_episodes']),
                'mean_raw': avg_metrics['avg_scores'],
                'std_raw': np.sqrt(avg_metrics['score_variance']),
                'mean_ma': avg_metrics['avg_mean_scores'],
                'std_ma': np.sqrt(avg_metrics['mean_score_variance'])
            })
            
            # Clear memory between runs
            del trainer
    finally:
        env.close()
    
    # Save results efficiently
    with open('ql_training_results.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return results

def plot_results(results):
    """Create 2x3 grid plot from training results"""
    plt.figure(figsize=(18, 12))
    
    # Add main title for the entire figure
    plt.suptitle('Cartpole_Q_Learning_Plots', fontsize=16)  # y adjusts vertical position
    
    # First row: Raw episodic returns
    for i, result in enumerate(results, 1):
        ax = plt.subplot(2, 3, i)
        ax.plot(result['episodes'], result['mean_raw'], label='Mean', color='tab:blue')
        ax.fill_between(result['episodes'],
                       result['mean_raw'] - result['std_raw'],
                       result['mean_raw'] + result['std_raw'],
                       alpha=0.2, color='tab:blue', label='±1 Std Dev')
        ax.set_title(f'Config {i} - Episodic Returns', fontsize=12)
        ax.set_xlabel('Episodes', fontsize=10)
        ax.set_ylabel('Return', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Second row: Moving averages
    for i, result in enumerate(results, 4):
        ax = plt.subplot(2, 3, i)
        ax.plot(result['episodes'], result['mean_ma'], label='100 episode MA', color='tab:orange')
        ax.fill_between(result['episodes'],
                       result['mean_ma'] - result['std_ma'],
                       result['mean_ma'] + result['std_ma'],
                       alpha=0.2, color='tab:orange', label='±1 Std Dev')
        ax.set_title(f'Config {i-3} - Moving Average Episodic Returns', fontsize=12)
        ax.set_xlabel('Episodes', fontsize=10)
        ax.set_ylabel('MA Return', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout(pad=3.0)
    plt.savefig('cartpole_ql.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Check for cached results first
    results_file = Path('ql_training_results.pkl')
    if results_file.exists():
        print("Loading cached results...")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
    else:
        print("No cached results found, training...")
        results = train_configurations()
    
    plot_results(results)

if __name__ == "__main__":
    main()