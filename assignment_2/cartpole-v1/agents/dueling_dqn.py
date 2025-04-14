import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork1(nn.Module):

    def __init__(self, state_size, action_size, update_type, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork1, self).__init__()
        
        self.type = update_type
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        # Value head
        self.value_fc = nn.Linear(fc2_units,1)
        
        # Adavantage head
        self.advantage_fc = nn.Linear(fc2_units,action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value  = self.value_fc(x)
        advantage = self.advantage_fc(x)
        
        if self.type == 'avg':
          q_values =  value + (advantage - torch.mean(advantage, dim=1, keepdim=True))
          
        else:
          q_values =  value + (advantage - torch.max(advantage, dim=1, keepdim=True)[0])
        
        return q_values
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    
    
class Agent_DDQN():

    def __init__(self, state_size, action_size,update_type,device,lr,BUFFER_SIZE,BATCH_SIZE,GAMMA,UPDATE_EVERY):

        ''' Agent Environment Interaction '''
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.lr = lr
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.update_every = UPDATE_EVERY
        self.type = update_type

        ''' Q-Network '''
        self.qnetwork_local = QNetwork1(state_size, action_size,self.type).to(self.device)
        self.qnetwork_target = QNetwork1(state_size, action_size,self.type).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr= self.lr)

        ''' Replay memory '''
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, device=self.device)

        ''' Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets '''
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        ''' Save experience in replay memory '''
        self.memory.add(state, action, reward, next_state, done)

        ''' If enough samples are available in memory, get random subset and learn '''
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

        """ +Q TARGETS PRESENT """
        ''' Updating the Network every 'UPDATE_EVERY' steps taken '''
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        ''' Epsilon-greedy action selection (Already Present) '''
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        ''' Get max predicted Q values (for next states) from target model'''
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        ''' Compute Q targets for current states '''
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        ''' Get expected Q values from local model '''
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        ''' Compute loss '''
        loss = F.mse_loss(Q_expected, Q_targets)

        ''' Minimize the loss '''
        self.optimizer.zero_grad()
        loss.backward()

        ''' Gradiant Clipping '''
        """ +T TRUNCATION PRESENT """
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        
        
''' Defining DQN Algorithm '''


def train_dueling_dqn(run, env, agent, episodes=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, optimal_return_per_episode = 500):
    episodic_rewards = []
    episodic_regrets = []
    
    eps = eps_start

    for episode in range(episodes):
        state, _ = env.reset()
        reward_per_episode = 0
        done = False
        
        while not done:
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience in replay buffer
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            reward_per_episode += reward
            
        # decay epsilon
        eps = max(eps_end, eps_decay*eps) 
        
        # Logging
        episodic_regrets.append(optimal_return_per_episode - reward_per_episode)
        episodic_rewards.append(reward_per_episode)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episodic_rewards[-100:])
            print(f"Run {run+1}, Episode {episode+1}: Avg Reward (last 100) = {avg_reward:.2f}, Eps = {eps:.3f}")

    return episodic_rewards, episodic_regrets