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
        self.fc = nn.Linear(state_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc(state))
        x = F.softmax(self.out(x), dim=1)
        return x 


# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Linear(state_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1) 
        
    def forward(self, state):
        x = F.relu(self.fc(state))
        x = self.out(x)
        return x


# Compute discounted returns
def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns    

def train_reinforce(env, run, episodes, gamma, lr, hidden_size, device):
    
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, hidden_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episodic_rewards = []
    episodic_regrets = []

    optimal_return_per_episode = -100  # Adjust based on env

    for episode in range(episodes):
        try:
            state, _ = env.reset()
        except:
            state = env.reset()
        
        reward_per_episode = 0
        reward_per_step = []
        log_probs = []
        done = False
        
        while not done:
            
            # Move state to device before calling select_action
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

            # Temporarily override policy.select_action logic to support device
            probs = policy(state_tensor)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            log_probs.append(log_prob)
            reward_per_step.append(reward)
            reward_per_episode += reward

            state = next_state
            done = terminated or truncated

        # Compute returns and normalize
        returns = compute_returns(reward_per_step, gamma)
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = torch.sum(torch.stack([-log_prob * R for log_prob, R in zip(log_probs, returns)]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episodic_rewards.append(reward_per_episode)
        episodic_regrets.append(optimal_return_per_episode - reward_per_episode)

        print(f"Run: {run+1}, Episode: {episode+1}/{episodes}, Reward: {reward_per_episode:.2f}")

    return episodic_rewards, episodic_regrets


def train_reinforce_with_baseline(env, run, episodes, gamma, policy_lr, value_lr,
                                  hidden_size_policy, hidden_size_value, device):
    
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, hidden_size_policy).to(device)
    value_function = ValueNetwork(env.observation_space.shape[0], hidden_size_value).to(device)

    policy_optimizer = optim.Adam(policy.parameters(), lr=policy_lr)
    value_optimizer = optim.Adam(value_function.parameters(), lr=value_lr)

    episodic_rewards = []
    episodic_regrets = []

    optimal_return_per_episode = -100  # Adjust based on env

    for episode in range(episodes):
        try:
            state, _ = env.reset()
        except:
            state = env.reset()

        reward_per_episode = 0
        reward_per_step = []
        log_probs = []
        value_per_step = []
        done = False

        while not done:
            # Convert state to tensor on device for action selection
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

            # Select action using policy network
            probs = policy(state_tensor)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(device)

            current_value = value_function(state_tensor)
            next_value = value_function(next_state_tensor).detach()

            # TD(0) Target
            done_flag = float(terminated or truncated)
            target = reward + gamma * next_value * (1 - done_flag)
            value_loss = F.mse_loss(current_value, target)

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            log_probs.append(log_prob)
            reward_per_step.append(reward)
            value_per_step.append(current_value.squeeze(0))  # shape: [1] → []

            reward_per_episode += reward
            state = next_state
            done = terminated or truncated

        # Compute and normalize returns
        returns = compute_returns(reward_per_step, gamma)
        returns = torch.tensor(returns).float().to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        values = torch.stack(value_per_step)
        advantages = returns - values

        policy_loss = -torch.sum(torch.stack(log_probs) * advantages.detach())

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        episodic_rewards.append(reward_per_episode)
        episodic_regrets.append(optimal_return_per_episode - reward_per_episode)

        print(f"Run: {run+1}, Episode: {episode+1}/{episodes}, Reward: {reward_per_episode:.2f}")

    return episodic_rewards, episodic_regrets
