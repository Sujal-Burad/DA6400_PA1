import argparse
import gym
import numpy as np
import wandb
from q_table import Qtable
from sarsa_agent import SarasLearner
from q_agent import QLearner
import wandb

# Initialize W&B
wandb.login()

# Define hyperparameters and their ranges
parameters_dict = {
    'algorithm': {'values': ['sarsa']},
    'alpha': {'min': 0.01, 'max': 1.0},
    'epsilon': {'min': 0.01, 'max': 0.2},
    'epsilon_decay': {'min': 0.9, 'max': 0.99},
    'gamma': {'value': 0.99},
    'num_episodes': {'value': 10000},
    'seed': {'values': [10, 20, 30, 40, 50]},
    'render': {'value': False}
}

# Create sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'Episode_return',
        'goal': 'maximize',
        'name': 'Total_steps',
        'goal': 'maximize'
    },
    'parameters': parameters_dict
}
# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="sarsa_minigrid_hyperparameter_finetuning_final_minimizing_regret")

# Define training function
def train(config=None):
    # Initialize a new W&B run
    with wandb.init(config=config) as run:
        config = wandb.config
        
        # Set seed
        np.random.seed(config.seed)
        
        # Define the environment
        if config.render:
            env = gym.make('MiniGrid-Dynamic-Obstacles-5x5-v0', render_mode='human')
        else:
            env = gym.make('MiniGrid-Dynamic-Obstacles-5x5-v0')
        
        if config.algorithm == "sarsa":
            # Rename the run based on hyperparameters
            run.name = f"{config.algorithm}_alpha{config.alpha}_epsilon{config.epsilon}_epsilon_decay{config.epsilon_decay}_seed{config.seed}"

            # Define the q-table
            q_table = Qtable(env.observation_space, env.action_space)
            
            # Instantiate the sarsa learner agent
            sarsa_learner = SarasLearner(config.alpha, config.gamma, config.epsilon, q_table, env, config.seed, config.epsilon_decay)
            
            # Learn the q-table
            reward_list, total_steps_list = sarsa_learner.learn(config.num_episodes)
            
            for reward in reward_list:
                wandb.log({"Episode_return": reward})
            for step in total_steps_list:
                wandb.log({'Total_steps': step})
                
        # wandb.finish()

# Run the sweep agent
wandb.agent(sweep_id, train)