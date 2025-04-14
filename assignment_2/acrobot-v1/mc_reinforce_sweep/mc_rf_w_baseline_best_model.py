import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import torch.optim as optim
import datetime
import gymnasium as gym
import glob
import io
import base64
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np 
import matplotlib.pyplot as plt
import wandb
from agents.mc_reinforce import train_reinforce_with_baseline
import pickle

# Best set of hyperparamters 
# Cumulative Regret = 58288.8

hyper = {
    
    'lr' : 0.004,
    'hidden_size': 64
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining the env
env = gym.make('Acrobot-v1')

# Hyperparameters
episodes = 1000
gamma = 0.99

lr = hyper['lr']
hidden_size = hyper['hidden_size']

all_episodic_returns = []
all_episodic_regrets = []

no_runs = 5

for run in range(no_runs):
    
    episodic_rewards, episodic_regrets = train_reinforce_with_baseline(env, run, episodes, 
                                                gamma, lr, hidden_size, device,seed=42)
        
    all_episodic_returns.append(episodic_rewards)
    all_episodic_regrets.append(episodic_regrets) 
    
    
# Dumping lists into a pickle file for plotting purpose.
with open('best_config/MC_rf_w_bl_best_config.pkl', 'wb') as file:
    pickle.dump((all_episodic_returns, all_episodic_regrets), file)
