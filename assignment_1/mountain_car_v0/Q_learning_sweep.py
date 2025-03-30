import gymnasium as gym
import numpy as np 
import matplotlib.pyplot as plt
import wandb
from utils import Mountain_Car_Q_Learning_Agent
    
sweep_config = {
    
    'name': 'Q_learning_Mountain_Car',
    
    "method": "bayes",  # Bayesian Optimization
    
    "metric": {
        "name": "Final_Cumulative_Regret",
        "goal": "minimize"
        
    },  # Optimize final score
    
    "parameters": {
        
        "alpha": {"distribution": "uniform", "min": 0.01, "max": 0.5},
        "tau": {"distribution": "uniform", "min": 0.01, "max": 2.0},  # Temperature for softmax exploration
        "tau_decay": {"distribution": "uniform", "min": 0.99, "max": 0.9999},  # Decay factor for tau
        "bin_size": {
            "distribution": "q_log_uniform_values",  # Ensures discrete integer values
            "q": 1,
            "min": 10,
            "max": 50
        },
        "min_tau": {"distribution": "uniform", "min": 0.01, "max": 0.1}  # Minimum temperature
    }
}


def main():
    
    wandb.init(project="rl_a1",entity="da6400_rl")
    
    config = wandb.config 
    
    run_name = f"alpha-{config.alpha:0.4f}_tau-{config.tau:0.4f}_taudec-{config.tau_decay:0.4f}_bin-{config.bin_size:0.4f}_mintau-{config.min_tau:0.4f}"

    wandb.run.name = run_name
    wandb.run.save()
    
    # Defining the env
    env = gym.make("MountainCar-v0")
    
    # Hyperparameters
    gamma = 0.99
    episodes = 10000
    alpha = config.alpha
    tau = config.tau
    tau_decay = config.tau_decay
    bin_size = [int(config.bin_size),int(config.bin_size)]
    min_tau = config.min_tau

    all_episodic_rewards = []
    all_steps = []
    all_episodic_regrets = []
    
    no_runs = 5
    
    for run in range(no_runs):
        
        agent_Q_learning = Mountain_Car_Q_Learning_Agent(env,bin_size,alpha,gamma,tau,tau_decay,episodes,min_tau)

        episodic_rewards = []
        steps  = []
        optimal_return_per_episode = -100
        episodic_regrets =  []

        for episode in range(episodes):
            
            state = agent_Q_learning.discretize_state(env.reset()[0])
            action = agent_Q_learning.choose_action(state)
            done = False
            
            reward_per_episode = 0
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
                reward_per_episode += reward
                step_per_episode += 1
                
                done = terminated or truncated
                
            # Decaying the epsilon
            agent_Q_learning.decay_tau()
            
             # Print progress every 100 episodes
            if (episode+1) % 100 == 0:
                print(f"Run : {run+1}, Episode {episode + 1}/{episodes}, Episodic reward: {reward_per_episode:.4f}")
            
            regret_per_episode = optimal_return_per_episode - reward_per_episode

            episodic_regrets.append(regret_per_episode)        
            episodic_rewards.append(reward_per_episode)
            steps.append(step_per_episode)
            
        all_episodic_rewards.append(episodic_rewards)
        all_episodic_regrets.append(episodic_regrets)
        all_steps.append(steps)
    
    mean_rewards = np.mean(all_episodic_rewards,axis=0)
    mean_regrets = np.mean(all_episodic_regrets,axis = 0)
    
    for episode in range(len(mean_rewards)):
            
        wandb.log({
            'Episodes':episode,
            'Episodic_Reward': mean_rewards[episode],
            'Mean_Episodic_Reward' : np.mean(mean_rewards[max(0, episode-100):episode+1]), # Averaging over the past 100 episodes
            'Regret' : mean_regrets[episode],
            'Cumulative_Regret' : np.sum(mean_regrets[:episode+1]),
        })
            
    wandb.log({
        'Final_Cumulative_Regret': np.sum(mean_regrets)
        })
    
    wandb.finish()
    
if __name__ == "__main__":
    
    sweep_id = wandb.sweep(sweep_config, project="rl_a1",entity="da6400_rl")
    wandb.agent(sweep_id, function=main, count= 30)

        