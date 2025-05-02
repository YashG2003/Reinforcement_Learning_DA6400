import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import numpy as np

ACTION_ARROWS = {
    0: '↓',   # South
    1: '↑',   # North
    2: '→',   # East
    3: '←',   # West
    4: '⛟',   # Pickup
    5: '⛟',   # Dropoff
    None: '•' # Terminate / No-op
}


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



def plot_rewards(rewards, filename):
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Return")
    plt.title("SMDP Q-Learning: Return per Episode")
    plt.savefig(filename)
    plt.close()

# def plot_q_values(Q, filename="q_values.png"):
#     plt.imshow(Q, aspect='auto')
#     plt.colorbar()
#     plt.xlabel("Option")
#     plt.ylabel("State")
#     plt.title("Learned Q-values")
#     plt.savefig(filename)
#     plt.close()
