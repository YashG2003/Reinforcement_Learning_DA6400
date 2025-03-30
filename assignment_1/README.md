# Reinforcement Learning Assignment 1

This repository contains implementations of Q-Learning and SARSA algorithms for two Gymnasium environments: CartPole-v1 and MountainCar-v0.

## Roll No and Name

1. Roll No: `ME21B062`, Name: Yash Gawande

2. Roll No: `CH21B033`, Name: Sameer Deshpande

## Repository Structure

### Folders
- `cartpole_v1/`: Contains Q-Learning and SARSA implementations for CartPole environment
  - `agents/`: RL agent implementations
  - `best_params/`: Optimal hyperparameter configurations
  - `best_params_plots/`: Plotting scripts for best configurations
  - `trainers/`: Training scripts for each algorithm
  - `utils/`: Helper functions (discretization, exploration strategies)
  - `sweeps/`: Weights & Biases hyperparameter sweep configurations

- `mountain_car_v0/`: Contains Q-Learning and SARSA implementations for MountainCar environment
  - Core implementation python files with best models and sweep configurations
  - Plotting utilities for visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YashG2003/Reinforcement_Learning_DA6400.git
cd assignment_1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Code

### For CartPole-v1:
```bash
# To run Q-Learning with best configurations
python -m cartpole_v1.best_params.q_learning_plots

# To run SARSA with best configurations
python -m cartpole_v1.best_params.sarsa_plots

# To run hyperparameter sweeps (requires W&B account)
python -m cartpole_v1.sweeps.q_learning_sweep
python -m cartpole_v1.sweeps.sarsa_sweep
```

### For MountainCar-v0:
```bash
# To run Q-Learning with best configurations
python mountain_car_v0/Q_learning_best_model.py

# To run SARSA with best configurations
python mountain_car_v0/SARSA_best_model.py

# To generate plots
python mountain_car_v0/plotting.py

# To run hyperparameter sweeps (requires W&B account)
python mountain_car_v0/Q_learning_sweep.py
python mountain_car_v0/SARSA_sweep.py
```

## Key Files

### CartPole-v1 Files
- `q_learning_agent.py`: Q-Learning agent implementation with softmax policy for CartPole
- `sarsa_agent.py`: SARSA agent implementation with ε-greedy policy for CartPole  
- `discretizer.py`: Discretizes continuous CartPole state space into discrete bins
- `exploration.py`: Implements ε-greedy, softmax, and decay strategies for exploration
- `best_configs.py`: Stores optimized hyperparameters for both algorithms
- `q_learning_trainer.py`: Training pipeline for Q-Learning agent
- `sarsa_trainer.py`: Training pipeline for SARSA agent  
- `q_learning_sweep.py`: W&B hyperparameter sweep config for Q-Learning
- `sarsa_sweep.py`: W&B hyperparameter sweep config for SARSA
- `q_learning_plots.py`: Generates performance plots for Q-Learning
- `sarsa_plots.py`: Generates performance plots for SARSA

### MountainCar-v0 Files
- `utils.py`: Contains both Q-Learning and SARSA agents with state discretization for MountainCar
- `Q_learning_sweep.py`: W&B sweep configuration for MountainCar Q-Learning
- `SARSA_sweep.py`: W&B sweep configuration for MountainCar SARSA  
- `Q_learning_best_model.py`: Trains Q-Learning agent with optimized hyperparameters
- `SARSA_best_model.py`: Trains SARSA agent with optimized hyperparameters
- `plotting.py`: Visualizes training results for both algorithms
- `Q_learning_MC_best*.pkl`: Saved results for top Q-Learning configurations  
- `SARSA_MC_best*.pkl`: Saved results for top SARSA configurations
- `*_MC_Plots.png`: Sample output plots for MountainCar experiments

## Results
The implementations generate:
* Training curves (episodic returns, moving average returns, regret, cumulative regret)
* Performance metrics (cumulative regret)
* Hyperparameter optimization results (when using W&B)

## Notes
1. For Weights & Biases sweeps, you'll need to:
   * Create a free W&B account
   * Update the `entity` parameter in sweep files
   * Login using `wandb login`

2. The MountainCar implementations use custom discretization approaches suited for the environment's continuous state space.

3. Best hyperparameters found through extensive sweeps are included in the repository.