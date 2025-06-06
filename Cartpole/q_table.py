import numpy as np

def Qtable(state_space, action_space, bin_size=100):
    """
        Description       : This function defines the q-table for the cartpole-v1 environment.
        Args:
            state_space   : The state space of the environment.
            action_space  : The action space of the environment.
            bin_size      : The number of bins for discretizing the state space.
        Returns:
            q_table       : The q-table for the cartpole-v1 environment.
            bins          : The bins for discretizing the state space.
        Info:
        state_space Shape : (4,)
        state_space High  : [ 4.8   inf  0.42  inf]
        state_space Low   : [-4.8 - inf -0.42 -inf]
    """
    # define the bins
    bins = np.zeros((state_space.shape[0], bin_size))

    # initialize the bins
    bins[0] = np.linspace(-4.8, 4.8, bin_size)
    bins[1] = np.linspace(-4, 4, bin_size)
    bins[2] = np.linspace(-0.42, 0.42, bin_size)
    bins[3] = np.linspace(-4, 4, bin_size)

    # define the q-table
    q_table = np.zeros((bin_size, bin_size, bin_size, bin_size, action_space.n))

    return q_table, bins


def discretize_state(state_space, bins):
    """
        Description        : This function discretizes the state space.
        Args:
            state_space    : The state space of the environment.
            bins           : The bins for discretizing the state space.
        Returns:
            state_discrete : The discretized state space.
    """
    state_discrete = np.zeros(state_space.shape)

    for i in range(state_space.shape[0]):
        state_discrete[i] = np.digitize(state_space[i], bins[i])
        # Clamp to ensure it doesn't go out of bounds
        state_discrete[i] = np.clip(state_discrete[i], 0, len(bins[i]) - 1)

    # print("discrete state = ", state_discrete)

    return tuple(state_discrete.astype(int))