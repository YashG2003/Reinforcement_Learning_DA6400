# Reinforcement Learning Assignment 3

This repository contains implementations of SMDP Q-Learning and Intra-option Q-Learning algorithms for the Taxi-v3 environment from Gymnasium.

## Roll No and Name

1. Roll No: `ME21B062`, Name: Yash Gawande, Task: SMDP, Intra-Option, report writing
2. Roll No: `CH21B033`, Name: Sameer Deshpande, Task: SMDP, Intra-Option, report writing

## Repository Structure

### Folders
- `smdp/`: Contains SMDP Q-Learning implementation
  - `visualizations/`: Stores generated plots and Q-value visualizations
  - `main.py`: Main script for running SMDP Q-Learning with navigation options
  - `main2.py`: Main script for running SMDP Q-Learning with pickup/dropoff options
  - `options.py`: Option definitions for navigation tasks
  - `options2.py`: Option definitions for pickup/dropoff tasks
  - `smdp_qlearning.py`: SMDP Q-Learning agent implementation
  - `utils.py`: Visualization and utility functions

- `intra_option/`: Contains Intra-option Q-Learning implementation
  - Similar file structure as `smdp/` but with intra-option learning implementation
  - `intra_option_qlearning.py`: Intra-option Q-Learning agent implementation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YashG2003/Reinforcement_Learning_DA6400.git
cd assignment_3
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Code

### For SMDP Q Learning:
```bash
cd smdp

# Run with navigation options (Nav-R, Nav-G, Nav-Y, Nav-B, Pickup, Dropoff)
python main.py

# Run with alternative options (Pickup, Dropoff only)
python main2.py
```

### For Intra-Option Q Learning:
```bash
cd intra_option

# Run with navigation options (Nav-R, Nav-G, Nav-Y, Nav-B, Pickup, Dropoff)
python main.py

# Run with alternative options (Pickup, Dropoff only)
python main2.py
```

## Key Files

### SMDP Q-Learning Files
- `smdp_qlearning.py`: SMDP Q-Learning agent implementation
- `options.py`: Defines 6 options (4 navigation + pickup + dropoff)
- `options2.py`: Defines 2 options (pickup + dropoff)
- `utils.py`: Visualization functions for Q-values and reward curves
- `main.py/main2.py`: Training scripts for different option sets

### Intra-option Q-Learning Files
- `intra_option_qlearning.py`: Intra-option Q-Learning agent implementation
-  Similar option and utility files as SMDP version

## Results
The implementations generate:
* Training curves showing total returns per episode and moving average episodic returns.
* Visualization of best options at each grid cell given the passenger location and destination.
* Visualizations of learned Q-values for each option
* Comparison between different option sets
* Comparison between SMDP and Intra-option learning approaches

## Notes
* Generated plots are saved in the visualizations/ folder
* Hyperparameters (alpha, gamma, epsilon) can be adjusted in the agent initialization
* The environment uses Taxi-v3 from Gymnasium with default settings
* Two different option sets are provided for comparison:
    * Navigation to 4 destinations + pickup/dropoff (6 options)
    * Pickup/dropoff only (2 options)