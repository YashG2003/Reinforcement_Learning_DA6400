import gymnasium as gym
import numpy as np 
import matplotlib.pyplot as plt

from utils import Mountain_Car_SARSA_Agent

np.random.seed(0)
    
env = gym.make("MountainCar-v0")

print(env.observation_space)
print(env.action_space)

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.8
epsilon_decay = 0.999
episodes = 10000
bin_size = [30,30]
min_epsilon = 0.01

agent_SARSA = Mountain_Car_SARSA_Agent(env,bin_size,alpha,gamma,epsilon,epsilon_decay,episodes,min_epsilon)

episodic_rewards = []
steps  = []

for episode in range(episodes):
    
    state = agent_SARSA.discretize_state(env.reset()[0])
    action = agent_SARSA.choose_action(state)
    done = False
    
    episode_per_reward = 0
    step_per_episode = 0
    
    while not done:
        
        # Acting in the environment
        next_state,reward,terminated,truncated,_ = env.step(action)
        
        # Discretizing the next state
        next_state = agent_SARSA.discretize_state(next_state)
        
        # Choosing the next action
        next_action = agent_SARSA.choose_action(next_state)
        
        # Updating the Q-value
        agent_SARSA.update_q_value(state,action,reward,next_state,next_action)
        
        # Updating the state and action
        state = next_state
        action = next_action
        
        # Updating the episode reward
        episode_per_reward += reward
        step_per_episode += 1
        
        done = terminated or truncated
        
    # Decaying the epsilon
    agent_SARSA.decay_epsilon()
    
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


