import numpy as np

class EpsilonGreedyPolicy:
    """
        Description: This class implements the epsilon greedy policy.
        Args:
            q_table  : The q-table for the cartpole-v1 environment.
            env      : The cartpole-v1 environment.
    """
    def __init__(self, q_table, env):
        self.q_table = q_table
        self.env = env

    def get_action(self, epsilon, state):
        """
            Description: This function returns an action based on the epsilon greedy policy.
            Args:
                epsilon  : The exploration rate.
                state      : The current state.
            Returns:
                action     : The action.
        """
        self.epsilon = epsilon
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])
        return action

class SoftmaxPolicy:
    """
        Description: This class implements the SoftmaxPolicy.
        Args:
            q_table  : The q-table for the cartpole-v1 environment.
            env      : The cartpole-v1 environment.
    """
    def __init__(self, q_table, env):
        self.q_table = q_table
        self.env = env

    def get_action(self, tau, state):
        """
            Description: This function returns an action based on the Softmax policy.
            Args:
                tau      : The Temperature in softmax rate.
                state      : The current state.
            Returns:
                action     : The action.
        """
        self.tau = tau
        q_values = [self.q_table[state + (action,)] for action in range(self.env.action_space.n)]
        # Apply softmax formula
        max_q_value = np.max(q_values)
        q_values = q_values - max_q_value
        exp_q_values = np.exp(np.array(q_values) / self.tau)
        probabilities = exp_q_values / np.sum(exp_q_values)
        # print("probabilities = ", probabilities)
        probabilities = np.clip(probabilities, a_min=1e-20, a_max= 1 - 1e-20)
        action = np.random.choice(self.env.action_space.n, p=probabilities)

        return action