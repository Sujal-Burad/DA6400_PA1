
import numpy as np
from q_table import discretize_state
from policy import EpsilonGreedyPolicy

class SarasLearner:
    """
        Description: This class implements the SARSA algorithm.
        Args:
            alpha    : The learning rate.
            gamma    : The discount factor.
            epsilon  : The exploration rate.
            q_table  : The q-table for the cartpole-v1 environment.
            bins     : The bins for discretizing the state space.
            env      : The cartpole-v1 environment.
    """
    def __init__(self, alpha, gamma, epsilon, q_table, bins, env, seed, epsilon_decay):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = q_table
        self.env = env
        self.bins = bins
        self.seed = seed
        self.epsilon_decay = epsilon_decay
        self.policy = EpsilonGreedyPolicy(self.q_table, self.env)

    def compute_td_error(self, state, action, next_state, next_state_action, reward):
        """
            Description: This function computes the TD error.
            Args:
                state               : The current state.
                action              : The current action.
                next_state          : The next state.
                reward              : The reward.
                next_state_action   : The next state action
            Returns:
                td_error   : The TD error.
        """
        return reward + self.gamma * self.q_table[next_state + (next_state_action,)] - self.q_table[state + (action,)]

    def update_q_table(self, state, action, td_error):
        """
            Description: This function updates the q-table.
            Args:
                state      : The current state.
                action     : The current action.
                td_error   : The TD error.
                Equations:
                Q(s, a) <- Q(s, a) + alpha * (reward + gamma * Q(s', a') - Q(s, a))
        """
        self.q_table[state + (action,)] += self.alpha * td_error

    def learn(self, num_episodes, num_steps):
        """
            Description: This function implements the SARSA algorithm.
            Args:
                num_episodes : The number of episodes.
                num_steps    : The number of steps.
            Returns:
                reward_list  : The list of rewards.
        """
        reward_list = []
        for episode in range(num_episodes):
            # initialize the environment
            state, _ = self.env.reset(seed=self.seed)
            # discretize the state
            state_discrete = discretize_state(state, self.bins)
            # initialize the reward
            total_reward = 0

            epsilon = self.epsilon

            if episode % 100 == 0:
                # Decaying epsilon
                epsilon = epsilon * self.epsilon_decay


            for step in range(num_steps):
                # get the action
                action = self.policy.get_action(epsilon, state_discrete)

                # take the action
                next_state, reward, done = self.env.step(action)[:3]

                # discretize the next state
                next_state_discrete = discretize_state(next_state, self.bins)
                # get the next_action 
                next_state_action = self.policy.get_action(epsilon, next_state_discrete)
                # update the q-table
                td_error = self.compute_td_error(state_discrete, action, next_state_discrete, next_state_action, reward)
                self.update_q_table(state_discrete, action, td_error)
                # update the state
                state = next_state
                # update the state_discrete
                state_discrete = next_state_discrete
                # update the total reward
                total_reward += reward
                # check if the episode is finished
                if done or step == num_steps - 1:
                    # print the total reward
                    print("Episode: {}/{}, Total Reward: {}".format(episode + 1, num_episodes, total_reward))
                    # append the total reward
                    reward_list.append(total_reward)
                    break
        return reward_list