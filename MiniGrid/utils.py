
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import os

def plot_reward(reward_list_seed, num_episodes, path="./", file_name="average_reward.png"):
    """
        Description       : This function plots the average reward.
        Args:
            reward_list_seed    : The list of rewards for 5 seeds.
            num_episodes        : The number of episodes.
            path                : The path for saving the plot.
            filename            : The name of the plot.
    """
    # Calculate mean and variance across seeds for each episode
    mean_rewards = np.mean(reward_list_seed, axis=0)
    variance_rewards = np.var(reward_list_seed, axis=0)
    plt.plot(np.arange(0, num_episodes), mean_rewards, label='Mean Reward')
    
    # Plot variance as a shaded region around the mean
    plt.fill_between(np.arange(0, num_episodes), mean_rewards - np.sqrt(variance_rewards), 
                     mean_rewards + np.sqrt(variance_rewards), alpha=1.0, label='Variance', color='red')

    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig(os.path.join(path, file_name))
    plt.close()
