import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import torch
from agents.mc_reinforce import train_reinforce, train_reinforce_with_baseline

def run_multiple(train_func, num_runs=5):
    all_rewards = []
    all_regrets = []
    for run in range(num_runs):
        rewards, regrets = train_func(run)
        all_rewards.append(rewards)
        all_regrets.append(regrets)

    all_rewards = np.array(all_rewards)
    avg_rewards = np.mean(all_rewards, axis=0)
    reward_variance = np.var(all_rewards, axis=0)

    avg_ma_rewards = np.array([np.mean(avg_rewards[max(0, i-100):i+1]) for i in range(len(avg_rewards))])
    ma_variance = np.array([np.var([np.mean(run[max(0, i-100):i+1]) for run in all_rewards]) for i in range(len(avg_rewards))])

    return {
        'episodes': np.arange(len(avg_rewards)),
        'mean_raw': avg_rewards,
        'std_raw': np.sqrt(reward_variance),
        'mean_ma': avg_ma_rewards,
        'std_ma': np.sqrt(ma_variance)
    }

def train_reinforce_config(baseline=False, best_config=None):
    def trainer(run):
        env = gym.make('CartPole-v1')
        
        if baseline:
            rewards, regrets = train_reinforce_with_baseline(
                env, run, 
                episodes=best_config['num_episodes'],
                gamma=best_config['gamma'],
                lr=best_config['lr'],
                hidden_size=best_config['hidden_size'],
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        else:
            rewards, regrets = train_reinforce(
                env, run, 
                episodes=best_config['num_episodes'],
                gamma=best_config['gamma'],
                lr=best_config['lr'],
                hidden_size=best_config['hidden_size'],
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        
        env.close()
        return rewards, regrets
    return trainer

def train_configurations():
    # Best configurations from your sweep results
    best_config_no_baseline = {
        'lr': 0.0035604,         
        'hidden_size': 64,   
        'gamma': 0.99,
        'num_episodes': 1000
    }
    
    best_config_with_baseline = {
        'lr': 0.004,          
        'hidden_size': 64,   
        'gamma': 0.99,
        'num_episodes': 1000
    }

    print("Training REINFORCE without baseline...")
    trainer_no_baseline = train_reinforce_config(baseline=False, best_config=best_config_no_baseline)
    result1 = run_multiple(trainer_no_baseline)

    print("Training REINFORCE with baseline...")
    trainer_with_baseline = train_reinforce_config(baseline=True, best_config=best_config_with_baseline)
    result2 = run_multiple(trainer_with_baseline)

    results = [result1, result2]

    # Save results
    with open('reinforce_training_results.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return results

def plot_results(results):
    plt.figure(figsize=(14, 10))
    plt.suptitle('CartPole MC Reinforce Comparison', fontsize=16)

    labels = ['Without Baseline', 'With Baseline']
    colors = ['tab:blue', 'tab:orange']

    # Episodic Return plots
    for i in range(2):
        ax = plt.subplot(2, 2, i + 1)
        result = results[i]
        ax.plot(result['episodes'], result['mean_raw'], label='Mean', color=colors[i])
        ax.fill_between(result['episodes'],
                        result['mean_raw'] - result['std_raw'],
                        result['mean_raw'] + result['std_raw'],
                        alpha=0.2, color=colors[i], label='±1 Std Dev')
        ax.set_title(f'{labels[i]} - Episodic Returns', fontsize=12)
        ax.set_xlabel('Episodes', fontsize=10)
        ax.set_ylabel('Return', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Moving Average Return plots
    for i in range(2):
        ax = plt.subplot(2, 2, i + 3)
        result = results[i]
        ax.plot(result['episodes'], result['mean_ma'], label='100-episode MA', color=colors[i])
        ax.fill_between(result['episodes'],
                        result['mean_ma'] - result['std_ma'],
                        result['mean_ma'] + result['std_ma'],
                        alpha=0.2, color=colors[i], label='±1 Std Dev')
        ax.set_title(f'{labels[i]} - Moving Average Returns', fontsize=12)
        ax.set_xlabel('Episodes', fontsize=10)
        ax.set_ylabel('MA Return', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout(pad=3.0)
    plt.savefig('cartpole_reinforce_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    results_file = Path('reinforce_training_results.pkl')
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