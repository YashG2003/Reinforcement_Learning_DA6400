import gymnasium as gym
from smdp_qlearning import SMDPQAgent
from options import option_policy, OPTION_NAMES, decode_state
from utils import plot_rewards_1, visualize_all_options, visualize_best_options
import numpy as np
import os

def run_episode(env, agent):
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        option = agent.select_option(state)
        option_steps = 0
        reward_sum = 0
        curr_state = state

        # Execute the option until termination (beta)
        while True:
            action = option_policy(env, option, state)
            if action is None:
                break  # Option terminates
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward_sum += (agent.gamma ** option_steps) * reward
            steps += 1
            option_steps += 1
            total_reward += reward
            state = next_state
            done = terminated or truncated
            if done:
                break

        # Update Q-value after option termination
        agent.update(curr_state, option, reward_sum, state, option_steps)

    return total_reward, steps

def main():
    env = gym.make("Taxi-v3")
    n_states = env.observation_space.n
    n_options = len(OPTION_NAMES)
    agent = SMDPQAgent(n_states, n_options, alpha=0.1, gamma=0.9, epsilon=0.1)

    episodes = 10000
    rewards = []

    for ep in range(episodes):
        ep_reward, ep_steps = run_episode(env, agent)
        rewards.append(ep_reward)
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}: Reward = {ep_reward}")

    os.makedirs('visualizations', exist_ok=True)
    
    # Plot both raw rewards and moving average
    plot_rewards_1(rewards, window=100)
    
    visualize_all_options(agent.Q, env, option_policy, OPTION_NAMES, filename ='visualizations/Q_values_1.png')
    
    visualize_best_options(agent.Q, env, pass_idx=0, dest_idx=3, option_names=OPTION_NAMES, 
                      filename='visualizations/best_options_passR_destB_1.png')
    
    visualize_best_options(agent.Q, env, pass_idx=4, dest_idx=2, option_names=OPTION_NAMES, 
                      filename='visualizations/best_options_passinT_destY_1.png')
    
    # Calculate and print average reward over last 100 episodes
    avg_reward = np.mean(rewards[-100:])
    print(f"Average reward over last 100 episodes: {avg_reward}")

if __name__ == "__main__":
    main()
