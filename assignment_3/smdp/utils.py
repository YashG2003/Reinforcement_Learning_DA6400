import matplotlib.pyplot as plt

def plot_rewards(rewards, filename="rewards.png"):
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SMDP Q-Learning: Reward per Episode")
    plt.savefig(filename)
    plt.close()

def plot_q_values(Q, filename="q_values.png"):
    plt.imshow(Q, aspect='auto')
    plt.colorbar()
    plt.xlabel("Option")
    plt.ylabel("State")
    plt.title("Learned Q-values")
    plt.savefig(filename)
    plt.close()
