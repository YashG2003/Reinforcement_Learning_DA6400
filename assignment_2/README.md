# Reinforcement Learning Assignment 2

This repository contains implementations of Dueling DQN and REINFORCE algorithms for Gymnasium environments: CartPole-v1 and Acrobot-v1.

## Roll No and Name

1. Roll No: `ME21B062`, Name: Yash Gawande

2. Roll No: `CH21B033`, Name: Sameer Deshpande

## Repository Structure

### Folders
- `cartpole-v1/`: Contains Dueling DQN and REINFORCE implementations for CartPole environment
  - `agents/`: RL agent implementations
  - `best_params_plots/`: Plots for best configurations
  - `cartpole_wandb_plots/`: Wandb hyperparameter tuning plots 
  - `plotting_code/`: code for generating plots of best configurations
  - `sweep_code/`: code for running Weights & Biases hyperparameter sweeps

- `acrobot-v1/`: Contains Dueling DQN and REINFORCE implementations for Acrobot environment
  - `agents/`: RL agent implementations
  - `best_config/`: contains pickle files having results of best configurations
  - `duel_dqn_sweep/`: code for running Weights & Biases hyperparameter sweeps
  - `mc_reinforce_sweep/`: code for running Weights & Biases hyperparameter sweeps
  - `top_params_plots/`: Plots for best configurations
  - `wandb_sweep_plots/`: Wandb hyperparameter tuning plots 
  

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YashG2003/Reinforcement_Learning_DA6400.git
cd assignment_2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Code

### For CartPole-v1:
```bash
cd cartpole-v1

# To run ddqn with best configurations
python -m plotting_code.ddqn_plots

# To run mcr with best configurations
python -m plotting_code.mcr_plots

# To run hyperparameter sweeps (requires W&B account)
python -m sweep_code.dueling_dqn_sweep_type1
python -m sweep_code.mc_reinforce_sweep
```

### For MountainCar-v0:
```bash
cd acrobot-v1

# To run ddqn with best configurations
python -m duel_dqn_sweep.duel_dqn_best_model_best_model_type1
python -m duel_dqn_sweep.duel_dqn_best_model_best_model_type2

# To run mcr with best configurations
python -m mc_reinforce_sweep.mc_rf_w_baseline_best_model
python -m mc_reinforce_sweep.mc_rf_without_baseline_best_model

# To run hyperparameter sweeps (requires W&B account)
python -m duel_dqn_sweep.duel_dqn_sweep_type1
python -m mc_reinforce_sweep.mc_reinforce_baseline_sweep
```

## Key Files

### CartPole-v1 Files
- `dueling_dqn.py`: Dueling DQN agent implementation for CartPole
- `mc_reinforce.py`: MC Reinforce agent implementation for CartPole  
- `ddqn_plots.py`: Plotting code for best configuration of DDQN
- `mcr_plots.py`: Plotting code for best configuration of MC reinforce
- `sweep_code` files: to run W&B hyperparameter sweeps
- `*training_reslts.pkl`: files having training results for best hyperparmaeters

### Acrobot-v1 Files
- `duel_dqn.py`: Dueling DQN agent implementation for CartPole
- `mc_reinforce.py`: MC Reinforce agent implementation for CartPole 
- `sweep_code` files: to run W&B hyperparameter sweeps
- `plotting.py`: Visualizes training results for both algorithms
- `*best_config.pkl`: files having training results for best hyperparmaeters

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

2. Best hyperparameters found through extensive sweeps are included in the repository.