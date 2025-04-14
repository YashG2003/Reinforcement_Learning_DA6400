import torch
import gymnasium as gym
import torch
import gymnasium as gym
import numpy as np 
import wandb
from agents.dueling_dqn import Agent_DDQN, train_dueling_dqn

    
sweep_config = {
    
    'name' : 'DDQN_cartpole_type1',
    
    "method": "bayes",  # Bayesian Optimization
    "metric": 
        {"name": "Final_Cumulative_Regret", 
         "goal": "minimize"},  # Optimize final score
        
    "parameters": {
            "lr": {'values': [1e-5, 5e-4, 1e-4]},
            "epsilon_decay": {'values': [0.993, 0.995]},
            "min_epsilon": {'values': [0.01, 0.02]},
            "update_every": {'values': [10, 20, 40]},
            "batch_size": {'values': [256, 512]}
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = int(1e6)  # replay buffer size
GAMMA = 0.99            # discount factor


def main():
    
    wandb.init(project="rl_a2",entity="da6400_rl")
    
    config = wandb.config 
    
    run_name = f"type_1_alpha_{config.lr}_batch_size_{config.batch_size}_epsdec_{config.epsilon_decay}_mineps_{config.min_epsilon}"

    wandb.run.name = run_name
    wandb.run.save()
    
    # Defining the env
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n
    optimal_return_per_episode = 500
    
    # Hyperparameters
    episodes = 2000
    lr = config.lr
    epsilon_decay = config.epsilon_decay
    min_epsilon = config.min_epsilon
    update_every = config.update_every
    batch_size = config.batch_size
    eps_start = 1
    
    # Update type 1
    update_type = 'avg'

    all_episodic_rewards = []
    all_episodic_regrets = []
    
    no_runs = 3
    
    for run in range(no_runs):
        
        agent_DDQN = Agent_DDQN(state_shape,action_shape, update_type,
                                device,lr,BUFFER_SIZE,batch_size,GAMMA,update_every)
        
        episodic_rewards, episodic_regrets = train_dueling_dqn(run,env,agent_DDQN,episodes, eps_start, 
                                                               min_epsilon, epsilon_decay, 
                                                               optimal_return_per_episode)
            
        all_episodic_rewards.append(episodic_rewards)
        all_episodic_regrets.append(episodic_regrets) 
               
        
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
    
    sweep_id = wandb.sweep(sweep_config, project="rl_a2",entity="da6400_rl")
    wandb.agent(sweep_id, function=main, count= 10)
