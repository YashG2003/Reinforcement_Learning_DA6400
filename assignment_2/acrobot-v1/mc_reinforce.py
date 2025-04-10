
import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()

        self.action_size = action_size
        self.fc = nn.Linear(state_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc(state))
        x = F.softmax(self.out(x), dim=1)
        return x 
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob
    
# Compute discounted rewards
def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns    


# Main MC REINFORCE algorithm

def train_reinforce(env,run, episodes, gamma, lr):
    
    policy = PolicyNetwork(state_size=env.observation_space.shape[0],
                           action_size=env.action_space.n)
    
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    episodic_rewards = []
    episodic_regrets =  []
    
    optimal_return_per_episode = -100

    for episode in range(episodes):
        
        state, _ = env.reset()
        
        reward_per_episode = 0
        
        reward_per_step = []
        
        done  = False
        
        log_probs = []
        
        while not done:
            
            action, log_prob = policy.select_action(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            
            log_probs.append(log_prob)
            
            reward_per_episode += reward
            
            reward_per_step.append(reward)
            
            state = next_state
            
            done = terminated or truncated
            
            if done:
                break

        returns = compute_returns(reward_per_step, gamma)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = 0
        
        for log_prob, R in zip(log_probs, returns):
            loss -= log_prob * R

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        regret_per_episode = optimal_return_per_episode - reward_per_episode

        episodic_regrets.append(regret_per_episode)        
        episodic_rewards.append(reward_per_episode)
        
        # Print progress every 1 episodes
        if (episode+1) % 1 == 0:
            print(f"Run : {run+1}, Episode {episode + 1}/{episodes}, Episodic reward: {reward_per_episode:.4f}")
            
    return episodic_rewards, episodic_regrets

    