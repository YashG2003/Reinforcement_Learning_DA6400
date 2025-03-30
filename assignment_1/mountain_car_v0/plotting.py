import matplotlib.pyplot as plt 
import pickle
import numpy as np

    
def moving_average(data, window=100):
    
    # Computing moving average with the given window size.
    return np.convolve(data, np.ones(window) / window, mode='valid')

def plot_results(returns_matrix_list, subtitle_name, save_path, window=100):
    """
    Plot episodic return and moving average episodic return across multiple configurations.

    Args:
        returns_matrix_list: List of numpy arrays (each of shape 5 x episodes) for different configurations.
        window: Window size for moving average.
    """
    num_configs = len(returns_matrix_list)  # Number of configurations
    fig, axes = plt.subplots(2, num_configs, figsize=(5 * num_configs, 8))  # 2 rows, num_configs cols

    for i, returns_matrix in enumerate(returns_matrix_list):
        episodes = np.arange(returns_matrix.shape[1])

        # Compute mean and variance across runs (axis=0)
        mean_return = np.mean(returns_matrix, axis=0)
        var_return = np.var(returns_matrix, axis=0)

        # Compute moving average and its variance
        mean_moving_avg = moving_average(mean_return, window)
        var_moving_avg = moving_average(var_return, window)
        episodes_moving_avg = np.arange(len(mean_moving_avg))

        # Plot raw episodic return
        ax1 = axes[0, i] if num_configs > 1 else axes[0]
        ax1.plot(episodes, mean_return, label='Mean', color='blue')
        ax1.fill_between(episodes, mean_return - np.sqrt(var_return), mean_return + np.sqrt(var_return), alpha=0.2, color='blue',label=r'$\pm1$ Std Dev')
        ax1.set_title(f'Config {i+1} - Episodic Returns')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Return')
        ax1.legend()

        # Plot moving average episodic return
        ax2 = axes[1, i] if num_configs > 1 else axes[1]
        ax2.plot(episodes_moving_avg, mean_moving_avg, label='100 episode MA', color='red')
        ax2.fill_between(episodes_moving_avg, mean_moving_avg - np.sqrt(var_moving_avg), mean_moving_avg + np.sqrt(var_moving_avg), alpha=0.2, color='red',label=r'$\pm1$ Std Dev')
        ax2.set_title(f'Config {i+1} - Moving Average Episodic Returns')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('MA Return')
        ax2.legend()
        
        
    plt.suptitle(subtitle_name, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


# Plotting the episodic_return and mean_episodic_return for SARSA on Mountain Car environment 

# Loading all lists from the pickle file
with open('SARSA_MC_best1_config.pkl', 'rb') as file:
    all_returns1, all_regrets1 = pickle.load(file)

with open('SARSA_MC_best2_config.pkl', 'rb') as file:
    all_returns2, all_regrets2 = pickle.load(file)
    
with open('SARSA_MC_best3_config.pkl', 'rb') as file:
    all_returns3, all_regrets3 = pickle.load(file)
    

# Episodic return data for 3 configurations (each is a 5 x 10000 matrix)
returns_matrix_list = [np.array(all_returns1), np.array(all_returns2), np.array(all_returns3)]

# Plotting the graphs
plot_results(returns_matrix_list,subtitle_name= 'Mountain_Car_SARSA_Plots',save_path = 'SARSA_MC_Plots.png',window=100)

    

    
# Plotting the episodic_return and mean_episodic_return for Q learning on Mountain Car environment 

# Loading all lists from the pickle file
with open('Q_learning_MC_best1_config.pkl', 'rb') as file:
    all_returns1, all_regrets1 = pickle.load(file)

with open('Q_learning_MC_best2_config.pkl', 'rb') as file:
    all_returns2, all_regrets2 = pickle.load(file)
    
with open('Q_learning_MC_best3_config.pkl', 'rb') as file:
    all_returns3, all_regrets3 = pickle.load(file)
    

# Episodic return data for 3 configurations (each is a 5 x 10000 matrix)
returns_matrix_list = [np.array(all_returns1), np.array(all_returns2), np.array(all_returns3)]

# Plotting the graphs
plot_results(returns_matrix_list,subtitle_name= 'Mountain_Car_Q_learning_Plots',save_path = 'Q_learning_MC_Plots.png',window=100)