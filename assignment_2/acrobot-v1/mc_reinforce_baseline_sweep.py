import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import torch.optim as optim
import datetime
import gymnasium as gym
from gym.wrappers.record_video import RecordVideo
import glob
import io
import base64
from IPython.display import HTML
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np 
import matplotlib.pyplot as plt
import wandb
from mc_reinforce import train_reinforce_with_baseline

sweep_config = {
    
    'name' : 'MC_Reinforce_with_baseline_Acrobot',
    
    "method": "bayes",  # Bayesian Optimization
    
    "metric": 
        {"name": "Final_Cumulative_Regret", 
         "goal": "minimize"},  # Optimize final score
        
    "parameters": {
        
        "policy_lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-3},
        "value_lr" : {"distribution": "uniform", "min": 1e-3, "max": 1e-2},
        
        "hidden_size_policy": {
            "values": [64, 128, 256]
    },  
        "hidden_size_value": {
            "values": [64, 128, 256]
    }
}
    
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    
    wandb.init(project="rl_a2",entity="da6400_rl")
    
    config = wandb.config 
    
    run_name = f"RL_sweep_with_baseline_plr-{config.policy_lr:0.4f}_vlr_{config.value_lr:0.4f}_phs_{config.hidden_size_policy}_vhs_{config.hidden_size_value}"

    wandb.run.name = run_name
    wandb.run.save()
    
    # Defining the env
    env = gym.make('Acrobot-v1')
    
    # Hyperparameters
    episodes = 1000
    gamma = 0.99
    
    policy_lr = config.policy_lr
    value_lr = config.value_lr
    hidden_size_policy = config.hidden_size_policy
    hidden_size_value = config.hidden_size_value
    
    all_episodic_rewards = []
    all_episodic_regrets = []
    
    no_runs = 5
    
    for run in range(no_runs):
        
        episodic_rewards, episodic_regrets = train_reinforce_with_baseline(env,run,
                                            episodes, gamma, policy_lr,value_lr,
                                         hidden_size_policy, hidden_size_value,device)
            
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
    wandb.agent(sweep_id, function=main, count= 30)

