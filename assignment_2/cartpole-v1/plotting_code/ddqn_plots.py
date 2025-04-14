import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from agents.dueling_dqn import train_dueling_dqn, Agent_DDQN

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

def train_dueling_dqn_config(update_type, best_config):
    def trainer(run):
        env = gym.make("CartPole-v1")
        state_shape = env.observation_space.shape[0]
        action_shape = env.action_space.n

        agent_DDQN = Agent_DDQN(state_shape, action_shape, update_type, 
                                device="cpu", 
                                lr=best_config['lr'],
                                BUFFER_SIZE=best_config['buffer_size'],
                                BATCH_SIZE=best_config['batch_size'],
                                GAMMA=best_config['gamma'],
                                UPDATE_EVERY=best_config['update_every'])

        rewards, regrets = train_dueling_dqn(run, env, agent_DDQN, 
                                             episodes=best_config['num_episodes'],
                                             eps_start=1.0, 
                                             eps_end=best_config['min_epsilon'],
                                             eps_decay=best_config['epsilon_decay'],
                                             optimal_return_per_episode=500)
        env.close()
        return rewards, regrets

    return trainer

def train_configurations():
    best_config_avg = {
        'lr': 1e-4,
        'batch_size': 512,
        'min_epsilon': 0.02,
        'epsilon_decay': 0.995,
        'update_every': 40,
        'buffer_size': int(1e6),
        'gamma': 0.99,
        'num_episodes': 1000
    }
    
    best_config_max = {
        'lr': 1e-4,
        'batch_size': 512,
        'min_epsilon': 0.02,
        'epsilon_decay': 0.995,
        'update_every': 40,
        'buffer_size': int(1e6),
        'gamma': 0.99,
        'num_episodes': 1000
    }    

    print("Training type 1 (avg update)...")
    trainer_1 = train_dueling_dqn_config('avg', best_config_avg)
    result1 = run_multiple(trainer_1)

    print("Training type 2 (max update)...")
    trainer_2 = train_dueling_dqn_config('max', best_config_max)
    result2 = run_multiple(trainer_2)

    results = [result1, result2]

    # Save results
    with open('ddqn_training_results.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return results

def plot_results(results):
    plt.figure(figsize=(14, 10))
    plt.suptitle('Cartpole Dueling DQN Comparison', fontsize=16)

    labels = ['Type 1 (avg)', 'Type 2 (max)']
    colors = ['tab:blue', 'tab:green']

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
    plt.savefig('cartpole_dueling_dqn.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    results_file = Path('ddqn_training_results.pkl')
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
