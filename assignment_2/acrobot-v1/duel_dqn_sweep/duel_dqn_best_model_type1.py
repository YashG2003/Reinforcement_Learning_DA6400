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
import pickle

 # Best set of hyperparameters
 # Cumulative Regret = 8075.2
 
hyper = {
     
     'lr' : 0.001,
     'epsilon_decay' : 0.9901,
     'min_epsilon' : 0.0028
 }


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
UPDATE_EVERY = 20       # how often to update the network (When Q target is present)

# Defining the env
env = gym.make('Acrobot-v1')
state_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

# Hyperparameters
episodes = 1000
epsilon = 1.0

# Configuration parameters
lr = hyper['lr']
epsilon_decay = hyper['epsilon_decay']
min_epsilon = hyper['min_epsilon']

# Update type 1
update_type = 'avg'

all_episodic_returns = []
all_episodic_regrets = []

no_runs = 5

for run in range(no_runs):
    
    agent_DDQN = Agent_DDQN(state_shape,action_shape, update_type,
                            device,lr,BUFFER_SIZE,BATCH_SIZE,GAMMA,UPDATE_EVERY)
    
    episodic_rewards, episodic_regrets = train_dueling_dqn(run,env,agent_DDQN,episodes, epsilon, min_epsilon, epsilon_decay)
        
    all_episodic_returns.append(episodic_rewards)
    all_episodic_regrets.append(episodic_regrets) 
    
# Dumping lists into a pickle file for plotting purpose.
with open('best_config/Duel_DQN_t1_best_config.pkl', 'wb') as file:
    pickle.dump((all_episodic_returns, all_episodic_regrets), file)