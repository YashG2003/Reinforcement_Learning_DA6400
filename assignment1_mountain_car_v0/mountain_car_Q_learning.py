import gymnasium as gym
import numpy as np 
import matplotlib.pyplot as plt

from utils import Mountain_Car_Q_Learning_Agent

np.random.seed(0)
    
env = gym.make("MountainCar-v0")

# Hyperparameters
alpha = 0.1
gamma = 0.99
tau = 0.8
tau_decay = 0.999
episodes = 10000
bin_size = [30,30]
min_tau = 0.01

agent_Q_learning = Mountain_Car_Q_Learning_Agent(env,bin_size,alpha,gamma,tau,tau_decay,episodes,min_tau)

episodic_rewards = []
steps  = []

for episode in range(episodes):
    
    state = agent_Q_learning.discretize_state(env.reset()[0])
    action = agent_Q_learning.choose_action(state)
    done = False
    
    episode_per_reward = 0
    step_per_episode = 0
    
    while not done:
        
        # Acting in the environment
        next_state,reward,terminated,truncated,_ = env.step(action)
        
        # Discretizing the next state
        next_state = agent_Q_learning.discretize_state(next_state)
        
        # Choosing the next action
        next_action = agent_Q_learning.choose_action(next_state)
        
        # Updating the Q-value
        agent_Q_learning.update_q_value(state, action, reward, next_state)
        
        # Updating the state and action
        state = next_state
        action = next_action
        
        # Updating the episode reward
        episode_per_reward += reward
        step_per_episode += 1
        
        done = terminated or truncated
        
    # Decaying the epsilon
    agent_Q_learning.decay_tau()
    
    episodic_rewards.append(episode_per_reward)
    steps.append(step_per_episode)
        
    # Print progress every 100 episodes
    if (episode+1) % 100 == 0:
        print(f"Episode {episode + 1}/{episodes}, Episode reward: {episode_per_reward:.4f}")
        
# Plot the rewards and steps
plt.figure(figsize=(12,6))
plt.plot(episodic_rewards)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards over episodes")
plt.show()

