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
    
    'name' : 'MC_RF_with_bl_Cartpole_Higher',
    
    "method": "grid",  # Bayesian Optimization
    
    "metric": 
        {"name": "Final_Cumulative_Regret", 
         "goal": "minimize"},  # Optimize final score
        
    "parameters": {
        
        "lr": {'values': [0.01, 0.005,0.004,0.008,0.006]},
        
        "hidden_size": {
            "values": [64,128]
    },  
}
    
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    
    wandb.init(project="rl_a2",entity="da6400_rl")
    
    config = wandb.config 
    
    run_name = f"RL_sweep_1_with_baseline_lr-{config.lr:0.4f}_hs_{config.hidden_size}"

    wandb.run.name = run_name
    wandb.run.save()
    
    # Defining the env
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    
    # Hyperparameters
    episodes = 1000
    gamma = 0.99
    
    lr = config.lr
    hidden_size = config.hidden_size
    
    all_episodic_rewards = []
    all_episodic_regrets = []
    
    no_runs = 5
    
    for run in range(no_runs):
        
        episodic_rewards, episodic_regrets = train_reinforce_with_baseline(env, run, episodes, 
                                                    gamma, lr, hidden_size, device,seed=42)
            
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
    wandb.agent(sweep_id, function=main, count = 10)

