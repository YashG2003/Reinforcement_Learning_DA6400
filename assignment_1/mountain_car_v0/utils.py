import numpy as np 
import gymnasium as gym

class Mountain_Car_SARSA_Agent:
    
    def __init__(self,
                 env,bin_size=[30,30],alpha=0.1,gamma=0.99,epsilon=0.1,
                 epsilon_decay=0.999,episodes=10000,min_epsilon=0.01):
        self.env = env
        self.bin_size = bin_size
        self.q_value = np.zeros((self.bin_size[0],self.bin_size[1],self.env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.discretize_bins = self.create_discretize_bins()
        
    def choose_action(self,state):
        
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.q_value[state[0],state[1]])
    
    def create_discretize_bins(self):
        
        grid = []
        
        for i in range(len(self.env.observation_space.high)):
            grid.append(np.linspace(self.env.observation_space.low[i],self.env.observation_space.high[i],self.bin_size[i]+1)[1:-1])
            
        return grid
    
    def discretize_state(self,state):
        
        state_index = []
        
        for i in range(len(state)):
            state_index.append(np.digitize(state[i],self.discretize_bins[i]))
            
        return tuple(state_index)
    
    def update_q_value(self,state,action,reward,next_state,next_action):
        
        self.q_value[state[0],state[1],action] += self.alpha*(reward+
                                                  self.gamma*self.q_value[next_state[0],next_state[1],next_action]
                                                - self.q_value[state[0],state[1],action])
        
    def decay_epsilon(self):
        
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.min_epsilon)
        
class Mountain_Car_Q_Learning_Agent:
    
    def __init__(self,
                 env,bin_size=[30,30],alpha=0.1,gamma=0.99,tau=0.5,
                 tau_decay=0.999,episodes=10000,min_tau=0.01):
        self.env = env
        self.bin_size = bin_size
        self.q_value = np.zeros((self.bin_size[0],self.bin_size[1],self.env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.episodes = episodes
        self.tau_decay = tau_decay
        self.min_tau = min_tau
        self.discretize_bins = self.create_discretize_bins()
        
    def choose_action(self,state):
        
        q_max = max(self.q_value[state[0],state[1]])
        q_value_exp = np.exp((self.q_value[state[0],state[1]]-q_max)/self.tau)
        
        q_value_prob = q_value_exp/np.sum(q_value_exp)
        
        return np.random.choice(self.env.action_space.n, p=q_value_prob)
    
    def create_discretize_bins(self):
        
        grid = []
        
        for i in range(len(self.env.observation_space.high)):
            grid.append(np.linspace(self.env.observation_space.low[i],self.env.observation_space.high[i],self.bin_size[i]+1)[1:-1])
            
        return grid
    
    def discretize_state(self,state):
        
        state_index = []
        
        for i in range(len(state)):
            state_index.append(np.digitize(state[i],self.discretize_bins[i]))
            
        return tuple(state_index)
    
    def update_q_value(self,state,action,reward,next_state):
        
        self.q_value[state[0],state[1],action] += self.alpha*(reward+
                                                  self.gamma*np.max(self.q_value[next_state[0],next_state[1]])
                                                - self.q_value[state[0],state[1],action])
        
    def decay_tau(self):
        
        self.epsilon = max(self.tau*self.tau_decay, self.min_tau)
        
        
        
        
        
        


