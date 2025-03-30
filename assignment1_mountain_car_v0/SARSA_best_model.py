import gymnasium as gym
import numpy as np 
import matplotlib.pyplot as plt
import wandb
from utils import Mountain_Car_SARSA_Agent
import pickle

# Function that computes mean_episodic_return and mean_episodic_regret for best set of hyperparameters
def train(env,bin_size,alpha,gamma,epsilon,epsilon_decay,episodes,min_epsilon):

    all_episodic_returns= []
    all_steps = []
    all_episodic_regrets = []

    no_runs = 5

    for run in range(no_runs):
        
        agent_SARSA = Mountain_Car_SARSA_Agent(env,bin_size,alpha,gamma,epsilon,epsilon_decay,episodes,min_epsilon)

        episodic_returns = []
        steps  = []
        optimal_return_per_episode = -100
        episodic_regrets =  []

        for episode in range(episodes):
            
            state = agent_SARSA.discretize_state(env.reset()[0])
            action = agent_SARSA.choose_action(state)
            done = False
            
            return_per_episode = 0
            step_per_episode = 0
            
            while not done:
                
                # Acting in the environment
                next_state,reward,terminated,truncated,_ = env.step(action)
                
                # Discretizing the next state
                next_state = agent_SARSA.discretize_state(next_state)
                
                # Choosing the next action
                next_action = agent_SARSA.choose_action(next_state)
                
                # Updating the action-value function
                agent_SARSA.update_q_value(state,action,reward,next_state,next_action)
                
                # Updating the state and action
                state = next_state
                action = next_action
                
                # Updating the episode reward
                return_per_episode += reward
                step_per_episode += 1
                
                done = terminated or truncated
                
            # Decaying the epsilon
            agent_SARSA.decay_epsilon()
            
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

# Cumulative regret1  = 521941.2
hyper1 = {
        
    "alpha": 0.198,
    "epsilon": 0.168,
    "epsilon_decay": 0.997,
    "bin_size": 49,
    "min_epsilon": 0.094
    }


# Cumulative regret2 = 526772.8
hyper2 = {
    
    "alpha": 0.205,
    "epsilon": 0.637,
    "epsilon_decay": 0.995,
    "bin_size": 44,
    "min_epsilon": 0.099
}   


# Cumulative regret3 = 528264.2
hyper3 = {
    
    "alpha": 0.144,
    "epsilon": 0.214,
    "epsilon_decay": 0.994,
    "bin_size": 45,
    "min_epsilon": 0.097
}

# Defining the env
env = gym.make("MountainCar-v0")

# 1st set of Hyperparameters
gamma = 0.99
episodes = 10000
alpha = hyper1['alpha']
epsilon = hyper1['epsilon']
epsilon_decay = hyper1['epsilon_decay']
bin_size = [int(hyper1['bin_size']),int(hyper1['bin_size'])]
min_epsilon = hyper1['min_epsilon']

all_returns1, all_regrets1 = train(env,bin_size,alpha,gamma,epsilon,epsilon_decay,episodes,min_epsilon)

# Dumping lists into a pickle file for plotting purpose.
with open('SARSA_MC_best1_config.pkl', 'wb') as file:
    pickle.dump((all_returns1, all_regrets1), file)
print('\n1st config returns dumped in pickle file')


# 2nd set of Hyperparameters
gamma = 0.99
episodes = 10000
alpha = hyper2['alpha']
epsilon = hyper2['epsilon']
epsilon_decay = hyper2['epsilon_decay']
bin_size = [int(hyper2['bin_size']),int(hyper2['bin_size'])]
min_epsilon = hyper2['min_epsilon']

all_returns2, all_regrets2 = train(env,bin_size,alpha,gamma,epsilon,epsilon_decay,episodes,min_epsilon)

# Dumping lists into a pickle file for plotting purpose.
with open('SARSA_MC_best2_config.pkl', 'wb') as file:
    pickle.dump((all_returns2, all_regrets2), file)
print('\n2nd config returns dumped in pickle file')


# 3rd set of Hyperparameters
gamma = 0.99
episodes = 10000
alpha = hyper3['alpha']
epsilon = hyper3['epsilon']
epsilon_decay = hyper3['epsilon_decay']
bin_size = [int(hyper3['bin_size']),int(hyper3['bin_size'])]
min_epsilon = hyper3['min_epsilon']

all_returns3, all_regrets3 = train(env,bin_size,alpha,gamma,epsilon,epsilon_decay,episodes,min_epsilon)

# Dumping lists into a pickle file for plotting purpose.
with open('SARSA_MC_best3_config.pkl', 'wb') as file:
    pickle.dump((all_returns3, all_regrets3), file)
print('\n3rd config returns dumped in pickle file')





