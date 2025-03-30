import numpy as np
import wandb
from typing import Dict, Any
from utils.discretizer import StateDiscretizer
from utils.exploration import ExplorationStrategy
from agents.sarsa_agent import SARSAAgent

class SARSATrainer:
    def __init__(self, env, config: Dict[str, Any]):
        self.env = env
        self.config = config
        self.discretizer = StateDiscretizer(num_bins=config['num_bins'])
        self.agent = SARSAAgent(
            state_space_shape=tuple(len(b) for b in self.discretizer.bins),
            action_space_n=env.action_space.n
        )
    
    def train(self):
        """Run training with multiple runs and track metrics."""
        metrics = {
            'scores': np.zeros((self.config['num_runs'], self.config['num_episodes'])),
            'regrets': np.zeros((self.config['num_runs'], self.config['num_episodes'])),
            'cumulative_regrets': np.zeros((self.config['num_runs'], self.config['num_episodes'])),
            'mean_scores': np.zeros((self.config['num_runs'], self.config['num_episodes']))
        }
        
        for run in range(self.config['num_runs']):
            self._run_episodes(run, metrics)
        
        return self._compute_averages(metrics)
    
    def _run_episodes(self, run, metrics):
        """Run all episodes for a single run."""
        cumulative_regret = 0
        
        for episode in range(self.config['num_episodes']):
            epsilon = self._get_exploration_rate(episode)
            total_score = self._run_episode(epsilon)
            
            regret = 500 - total_score
            cumulative_regret += regret
            
            metrics['scores'][run, episode] = total_score
            metrics['regrets'][run, episode] = regret
            metrics['cumulative_regrets'][run, episode] = cumulative_regret
            metrics['mean_scores'][run, episode] = np.mean(
                metrics['scores'][run, max(0, episode-99):episode+1]
            )
    
    def _run_episode(self, epsilon):
        """Run a single episode."""
        state, _ = self.env.reset()
        state = self.discretizer.discretize(state)
        terminated = truncated = False
        total_score = 0
        
        # Select initial action
        action = self.agent.get_action(state, epsilon)
        
        while not (terminated or truncated):
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = self.discretizer.discretize(next_state)
            
            # Select next action (on-policy)
            next_action = self.agent.get_action(next_state, epsilon)
            
            # SARSA update
            self.agent.update(
                state, action, reward, next_state, next_action,
                self.config['gamma'], self.config['learning_rate']
            )
            
            total_score += reward
            state, action = next_state, next_action
        
        return total_score
    
    def _get_exploration_rate(self, episode):
        """Get exploration rate (epsilon) based on decay schedule."""
        if self.config['decay_type'] == 'linear':
            return ExplorationStrategy.linear_decay(
                self.config['epsilon_start'], self.config['epsilon_end'],
                episode, self.config['num_episodes']
            )
        elif self.config['decay_type'] == 'exponential':
            return ExplorationStrategy.exponential_decay(
                self.config['epsilon_start'], self.config['epsilon_end'],
                episode, self.config['decay_rate']
            )
        return self.config['epsilon_start']
    
    
    def _compute_averages(self, metrics):
        """Compute average metrics across runs with variance."""
        return {
            'avg_scores': np.mean(metrics['scores'], axis=0),
            'score_variance': np.var(metrics['scores'], axis=0),
            'avg_regrets': np.mean(metrics['regrets'], axis=0),
            'avg_mean_scores': np.mean(metrics['mean_scores'], axis=0),
            'mean_score_variance': np.var(metrics['mean_scores'], axis=0),
            'avg_cumulative_regrets': np.mean(metrics['cumulative_regrets'], axis=0),
            "Final_Cumulative_Regret": np.mean(metrics['cumulative_regrets'][:, -1])
        }