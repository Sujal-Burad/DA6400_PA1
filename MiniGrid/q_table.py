import numpy as np

# NO discretization of state is needed 

def Qtable(state_space, action_space):
    """
        Description       : This function defines the q-table for the cartpole-v1 environment.
        Args:
            state_space   : The state space of the environment.
            action_space  : The action space of the environment.
            bin_size      : The number of bins for discretizing the state space.
        Returns:
            q_table       : The q-table for the cartpole-v1 environment.
            # No bins, since the environment's state are itself discrete
        Info:
        state_space shape : (9, 4, 2)
        9 -> 9 grid cells
        4 -> direction of agent
        2 -> the immediate cell towards which the agent is pointing can be clear or have an obstacle (1 if not clear (not goal), else 0)
        action_space shape : (3) (turn left, turn right, move forward)
    """
    # define the q-table
    q_table = np.zeros((9, 4, 2, action_space.n))

    return q_table