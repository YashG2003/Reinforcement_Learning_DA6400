import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Circle

ACTION_ARROWS = {
    0: '↓',   # South
    1: '↑',   # North
    2: '→',   # East
    3: '←',   # West
    4: '⛟',   # Pickup
    5: '⛟',   # Dropoff
    None: '•' # Terminate / No-op
}

def plot_rewards_1(rewards, window=100):
    # Plot episodic rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episodic Returns')
    plt.xlabel('Episode')
    plt.ylabel('Total Return')
    plt.title('SMDP Q learning Episodic Returns')
    plt.legend()
    plt.savefig("visualizations/smdp_episodic_returns_1.png")
    plt.close()

    # Compute moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    else:
        moving_avg = np.array(rewards)  # If not enough episodes, just plot raw rewards

    # Plot moving average
    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg, color='orange', label=f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Returns')
    plt.title(f'SMDP Q learning Moving Average Returns (window={window})')
    plt.legend()
    plt.savefig("visualizations/smdp_moving_avg_returns_1.png")
    plt.close()


def plot_rewards_2(rewards, window=100):
    # Plot episodic rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episodic Returns')
    plt.xlabel('Episode')
    plt.ylabel('Total Returns')
    plt.title('SMDP Q learning Episodic Returns')
    plt.legend()
    plt.savefig("visualizations/smdp_episodic_returns_2.png")
    plt.close()

    # Compute moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    else:
        moving_avg = np.array(rewards)  # If not enough episodes, just plot raw rewards

    # Plot moving average
    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg, color='orange', label=f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Returns')
    plt.title(f'SMDP Q learning Moving Average Returns (window={window})')
    plt.legend()
    plt.savefig("visualizations/smdp_moving_avg_returns_2.png")
    plt.close()


def visualize_best_options(Q, env, pass_idx, dest_idx, option_names, filename=None):
    """
    Visualize the best options for each state in a 5x5 grid for a fixed passenger location and destination.
    
    Args:
        Q: Q-value table (states x options)
        env: Taxi environment
        pass_idx: Passenger location index (0-4, where 4 means passenger in taxi)
        dest_idx: Destination index (0-3)
        option_names: List of option names
        filename: Optional filename to save the plot
    """
    # Create a 5x5 grid to store best options and their Q-values
    best_options = np.zeros((5, 5), dtype=int)
    best_q_values = np.zeros((5, 5))
    
    # Find best option and Q-value for each cell
    for row in range(5):
        for col in range(5):
            state = env.unwrapped.encode(row, col, pass_idx, dest_idx)
            best_option = np.argmax(Q[state])
            best_options[row, col] = best_option
            best_q_values[row, col] = Q[state, best_option]
    
    # Scale Q-values for circle sizes with better min/max constraints
    min_q = np.min(best_q_values)
    max_q = np.max(best_q_values)
    
    # Avoid division by zero if all Q-values are the same
    if max_q == min_q:
        scaled_q_values = np.ones_like(best_q_values) * 0.3  # Medium size
    else:
        # Scale to a more reasonable range (0.15 to 0.4) to prevent circles from being too big or too small
        scaled_q_values = 0.15 + 0.25 * (best_q_values - min_q) / (max_q - min_q)
    
    # Create a colormap for options
    n_options = len(option_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_options))
    
    # Create figure and axis with smaller grid cells
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw grid
    for i in range(6):
        ax.axhline(i, color='black', lw=1)
        ax.axvline(i, color='black', lw=1)
    
    # Draw circles for each cell
    for row in range(5):
        for col in range(5):
            option = best_options[row, col]
            size = scaled_q_values[row, col]
            
            circle = Circle((col + 0.5, 4.5 - row), size/2, 
                           color=colors[option], alpha=0.7)
            ax.add_patch(circle)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                      label=option_names[i], markerfacecolor=colors[i], markersize=10) 
                      for i in range(n_options)]
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, 1.15), ncol=min(5, n_options))
    
    # Set title and labels
    passenger_loc = "in taxi" if pass_idx == 4 else f"at {['R', 'G', 'Y', 'B'][pass_idx]}"
    destination = ['R', 'G', 'Y', 'B'][dest_idx]
    ax.set_title(f"Best Options (Passenger {passenger_loc}, Destination {destination})")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    
    # Set axis limits and ticks
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks(np.arange(0.5, 5.5, 1))
    ax.set_yticks(np.arange(0.5, 5.5, 1))
    ax.set_xticklabels(range(5))
    ax.set_yticklabels(range(5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if filename provided
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()



def visualize_all_options(q_table, env,option_policy,OPTION_NAMES,filename):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    cmap = plt.cm.plasma
    

    # Precompute Q values for all grid positions and options
    all_q_vals = []
    option_grids = [{} for _ in range(6)]

    for option_idx in range(6):
        for row in range(5):
            for col in range(5):
                max_q = -np.inf
                best_action = None
                for pass_loc in range(5):
                    for dest in range(4):
                        state = env.unwrapped.encode(row, col, pass_loc, dest)
                        q_val = q_table[state, option_idx]
                        if q_val > max_q:
                            max_q = q_val
                            best_action = option_policy(env, option_idx, state)
                all_q_vals.append(max_q)
                option_grids[option_idx][(row, col)] = (max_q, best_action)

    # Normalize color across all subplots
    min_q, max_q = min(all_q_vals), max(all_q_vals)
    norm = colors.Normalize(vmin=min_q, vmax=max_q)

    for option_idx in range(6):
        ax = axes[option_idx // 3, option_idx % 3]
        ax.set_title(f"Option: {OPTION_NAMES[option_idx]}", fontsize=16)
        for row in range(5):
            for col in range(5):
                q_val, action = option_grids[option_idx][(row, col)]
                rect = patches.Rectangle((col, 4 - row), 1, 1,
                                         facecolor=cmap(norm(q_val)),
                                         edgecolor='black')
                ax.add_patch(rect)
                ax.text(col + 0.5, 4 - row + 0.5,
                        ACTION_ARROWS.get(action, '?'),
                        ha='center', va='center', fontsize=14, color='white')
    
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_xticks([])
        ax.set_yticks([])

    # Add shared colorbar
    plt.suptitle('Visualization of learned Q values')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_label('Max Q-value across states')

    plt.savefig(filename)
    plt.close()
    
def visualize_all_options_2(q_table, env, option_policy, OPTION_NAMES, filename):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib import colors

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.cm.plasma

    # Precompute Q values
    all_q_vals = []
    option_grids = [{} for _ in range(2)]

    for option_idx in range(2):  # Only first 2 options
        for row in range(5):
            for col in range(5):
                max_q = -np.inf
                best_action = None
                for pass_loc in range(5):
                    for dest in range(4):
                        state = env.unwrapped.encode(row, col, pass_loc, dest)
                        q_val = q_table[state, option_idx]
                        if q_val > max_q:
                            max_q = q_val
                            best_action = option_policy(env, option_idx, state)
                all_q_vals.append(max_q)
                option_grids[option_idx][(row, col)] = (max_q, best_action)

    # Normalize across both plots
    min_q, max_q = min(all_q_vals), max(all_q_vals)
    norm = colors.Normalize(vmin=min_q, vmax=max_q)

    for option_idx in range(2):
        ax = axes[option_idx]
        ax.set_title(f"Option: {OPTION_NAMES[option_idx]}", fontsize=16)
        for row in range(5):
            for col in range(5):
                q_val, action = option_grids[option_idx][(row, col)]
                rect = patches.Rectangle((col, 4 - row), 1, 1,
                                         facecolor=cmap(norm(q_val)),
                                         edgecolor='black')
                ax.add_patch(rect)
                ax.text(col + 0.5, 4 - row + 0.5,
                        ACTION_ARROWS.get(action, '?'),
                        ha='center', va='center', fontsize=14, color='white')


        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colorbar
    plt.suptitle('Visualization of Q values for 2 options')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_label('Max Q-value across states')

    plt.savefig(filename)
    plt.close()
