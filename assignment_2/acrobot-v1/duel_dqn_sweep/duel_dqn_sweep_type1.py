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
from agents.duel_dqn import Agent_DDQN, train_dueling_dqn

    
sweep_config = {
    
    'name' : 'DDQN_Acrobot_Type1_No_Relu',
    
    "method": "bayes",  # Bayesian Optimization
    "metric": 
        {"name": "Final_Cumulative_Regret", 
         "goal": "minimize"},  # Optimize final score
        
    "parameters": {
        
        "lr": {'values': [0.001, 0.003,0.0001,0.0003]},
        "epsilon_decay": {"distribution": "uniform", "min": 0.99, "max": 0.999},
        "min_epsilon": {"distribution": "uniform", "min": 0.001, "max": 0.01}
    }
}

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
UPDATE_EVERY = 20       # how often to update the network (When Q target is present)


def main():
    
    wandb.init(project="rl_a2",entity="da6400_rl")
    
    config = wandb.config 
    
    run_name = f"RL_type1_sweep_2_lr-{config.lr:0.4f}_epsdec-{config.epsilon_decay:0.4f}_mineps-{config.min_epsilon:0.4f}"

    wandb.run.name = run_name
    wandb.run.save()
    
    # Defining the env
    env = gym.make('Acrobot-v1')
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n
    
    # Hyperparameters
    episodes = 1000
    epsilon = 1.0
    
    # Configuration parameters
    lr = config.lr
    epsilon_decay = config.epsilon_decay
    min_epsilon = config.min_epsilon
    
    # Update type 1
    update_type = 'avg'

    all_episodic_rewards = []
    all_episodic_regrets = []
    
    no_runs = 5
    
    for run in range(no_runs):
        
        agent_DDQN = Agent_DDQN(state_shape,action_shape, update_type,
                                device,lr,BUFFER_SIZE,BATCH_SIZE,GAMMA,UPDATE_EVERY)
        
        episodic_rewards, episodic_regrets = train_dueling_dqn(run,env,agent_DDQN,episodes, epsilon, min_epsilon, epsilon_decay)
            
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
    wandb.agent(sweep_id, function=main, count= 20)


