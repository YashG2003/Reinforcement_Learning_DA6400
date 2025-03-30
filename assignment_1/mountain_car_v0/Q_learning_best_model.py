import gymnasium as gym
import numpy as np 
import matplotlib.pyplot as plt
import wandb
from utils import Mountain_Car_Q_Learning_Agent
import pickle

# Function that computes mean_episodic_return and mean_episodic_regret for best set of hyperparameters
def train(env,bin_size,alpha,gamma,tau,tau_decay,episodes,min_tau):


    all_episodic_returns= []
    all_steps = []
    all_episodic_regrets = []

    no_runs = 5

    for run in range(no_runs):
        
        agent_Q_learning = Mountain_Car_Q_Learning_Agent(env,bin_size,alpha,gamma,tau,tau_decay,episodes,min_tau)

        episodic_returns = []
        steps  = []
        optimal_return_per_episode = -100
        episodic_regrets =  []

        for episode in range(episodes):
            
            state = agent_Q_learning.discretize_state(env.reset()[0])
            action = agent_Q_learning.choose_action(state)
            done = False
            
            return_per_episode = 0
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
                return_per_episode += reward
                step_per_episode += 1
                
                done = terminated or truncated
                
            # Decaying the epsilon
            agent_Q_learning.decay_tau()
            
            # Print progress every 100 episodes
            if (episode+1) % 100 == 0:
                print(f"Run : {run+1}, Episode {episode + 1}/{episodes}, Episodic return: {return_per_episode:.4f}")
            
            regret_per_episode = optimal_return_per_episode - return_per_episode

            episodic_regrets.append(regret_per_episode)        
            episodic_returns.append(return_per_episode)
            steps.append(step_per_episode)
            
            
        all_episodic_returns.append(episodic_returns)
        all_episodic_regrets.append(episodic_regrets)
        all_steps.append(steps)
    
    
    return all_episodic_returns, all_episodic_regrets


# Top 3 hyperparameters

# Cumulative regret1 = 272849.6
hyper1 = {
    
    "alpha": 0.3941,
    "tau": 0.1527,  # Temperature for softmax exploration
    "tau_decay": 0.9966,  # Decay factor for tau
    "bin_size": 48,
    "min_tau": 0.0505  # Minimum temperature
        
}

# Cumulative regret2 = 297827 
hyper2 = {
    
    "alpha": 0.2634,
    "tau": 0.1266,  # Temperature for softmax exploration
    "tau_decay": 0.9976,  # Decay factor for tau
    "bin_size": 46,
    "min_tau": 0.0960  # Minimum temperature
    
}

# Cumulative regret3 = 311842.4
hyper3 = {
    
    "alpha": 0.2314,
    "tau": 0.0121,  # Temperature for softmax exploration
    "tau_decay": 0.9945,  # Decay factor for tau
    "bin_size": 45,
    "min_tau": 0.0814  # Minimum temperature
}

# Defining the env
env = gym.make("MountainCar-v0")

# 1st set of Hyperparameters
gamma = 0.99
episodes = 10000
alpha = hyper1['alpha']
tau = hyper1['tau']
tau_decay = hyper1['tau_decay']
bin_size = [int(hyper1['bin_size']),int(hyper1['bin_size'])]
min_tau = hyper1['min_tau']

all_returns1, all_regrets1 = train(env,bin_size,alpha,gamma,tau,tau_decay,episodes,min_tau)

# Dumping lists into a pickle file for plotting purpose.
with open('Q_learning_MC_best1_config.pkl', 'wb') as file:
    pickle.dump((all_returns1, all_regrets1), file)
print('\n1st config returns dumped in pickle file')


# 2nd set of Hyperparameters
gamma = 0.99
episodes = 10000
alpha = hyper2['alpha']
tau = hyper2['tau']
tau_decay = hyper2['tau_decay']
bin_size = [int(hyper2['bin_size']),int(hyper2['bin_size'])]
min_tau = hyper2['min_tau']

all_returns2, all_regrets2 = train(env,bin_size,alpha,gamma,tau,tau_decay,episodes,min_tau)

# Dumping lists into a pickle file for plotting purpose.
with open('Q_learning_MC_best2_config.pkl', 'wb') as file:
    pickle.dump((all_returns2, all_regrets2), file)
print('\n2nd config returns dumped in pickle file')


# 3rd set of Hyperparameters
gamma = 0.99
episodes = 10000
alpha = hyper3['alpha']
tau = hyper3['tau']
tau_decay = hyper3['tau_decay']
bin_size = [int(hyper3['bin_size']),int(hyper3['bin_size'])]
min_tau = hyper3['min_tau']


all_returns3, all_regrets3 =  train(env,bin_size,alpha,gamma,tau,tau_decay,episodes,min_tau)

# Dumping lists into a pickle file for plotting purpose.
with open('Q_learning_MC_best3_config.pkl', 'wb') as file:
    pickle.dump((all_returns3, all_regrets3), file)
print('\n3rd config returns dumped in pickle file')





