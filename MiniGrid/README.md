For minigrid, I have also added a plot for total steps for each episode, which tells the number of steps for which the episode ran.

The total steps are positively correlated with the episodic return.

Note:
- **Number of bins is not needed here** as the state space is already discrete.
- **Number of steps is not needed here** as the environment gets terminated when it reaches a certain number of steps. (**Default max_steps = 100**)

### Command Line Arguments

The following command line arguments are available for configuring the environment and training process (when running the main.py file)


### Explanation of Arguments

- **`--algorithm`**: The reinforcement learning algorithm to use. Currently supports `sarsa` and `q_learning`.
- **`--alpha`**: The learning rate for updating the Q-values.
- **`--gamma`**: The discount factor for future rewards.
- **`--epsilon`**: The exploration rate for selecting random actions.
- **`--epsilon_decay`**: The decay rate for the exploration rate.
- **`--tau`**: The temperature parameter for softmax action selection.
- **`--tau_decay`**: The decay rate for the temperature parameter.
- **`--num_episodes`**: The total number of episodes to train.
- **`--seed`**: A list of seeds for reproducibility across different runs.
- **`--render`**: Whether to render the environment during training.
- **`--wandb_project_name`**: The name of the Weights & Biases project for logging results.


### Hyperparameter Configuration

The hyperparameters for the Q-learning algorithm are defined as follows (run hyperparameter_tuning_q_learning.py for this):

| Hyperparameter | Type | Range/Values |
|----------------|------|--------------|
| **algorithm**   | Categorical | `q_learning` |
| **alpha**       | Continuous | `[0.01, 1.0]` |
| **tau**         | Continuous | `[0.5, 4.0]` |
| **tau_decay**   | Continuous | `[0.9, 0.99]` |
| **gamma**       | Fixed      | `0.99` |
| **num_episodes**| Fixed      | `10000` |
| **seed**        | Categorical | `[10, 20, 30, 40, 50]` |
| **render**      | Fixed      | `False` |

The hyperparameters for the sarsa algorithm are defined as follows (run hyperparameter_tuning_sarsa.py for this):

| Hyperparameter | Type | Range/Values |
|----------------|------|--------------|
| **algorithm**   | Categorical | `q_learning` |
| **alpha**       | Continuous | `[0.01, 1.0]` |
| **epsilon**         | Continuous | `[0.01, 0.2]` |
| **epsilon_decay**   | Continuous | `[0.9, 0.99]` |
| **gamma**       | Fixed      | `0.99` |
| **num_episodes**| Fixed      | `10000` |
| **seed**        | Categorical | `[10, 20, 30, 40, 50]` |
| **render**      | Fixed      | `False` |


The wandb plots for SARSA are:
1. [5 seed runs](https://wandb.ai/sujal/minigrid_rl_experiment_sarsa_minimizing_regret)
2.  [Hyperparameter tuning sweep](https://wandb.ai/sujal/sarsa_minigrid_hyperparameter_finetuning_final_minimizing_regret/sweeps/y6ifv3kz)


The wandb plots for Q-learning are:
1. [5 seed runs](https://wandb.ai/sujal/minigrid_rl_experiment_q_learning_minimizing_regret)
2.  [Hyperparameter tuning sweep](https://wandb.ai/sujal/q_learning_minigrid_hyperparameter_finetuning_final_minimizing_regret/sweeps/tu8b7ieq)
