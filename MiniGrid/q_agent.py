
import numpy as np
# from q_table import discretize_state
from policy import SoftmaxPolicy

class QLearner:
    """
        Description: This class implements the Q-Learning algorithm.
        Args:
            alpha    : The learning rate.
            gamma    : The discount factor.
            tau  : The exploration rate.
            q_table  : The q-table for the cartpole-v1 environment.
            bins     : The bins for discretizing the state space.
            env      : The cartpole-v1 environment.
    """
    def __init__(self, alpha, gamma, tau, q_table, env, seed, tau_decay):
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.q_table = q_table
        self.env = env
        self.seed = seed
        self.tau_decay = tau_decay
        self.policy = SoftmaxPolicy(self.q_table, self.env)

    def compute_td_error(self, state, action, next_state, reward):
        """
            Description: This function computes the TD error.
            Args:
                state      : The current state.
                action     : The current action.
                next_state : The next state.
                reward     : The reward.
            Returns:
                td_error   : The TD error.
        """
        return reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state + (action,)]

    def update_q_table(self, state, action, td_error):
        """
            Description: This function updates the q-table.
            Args:
                state      : The current state.
                action     : The current action.
                td_error   : The TD error.
            Equations:
                Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a Q(s', a) - Q(s, a))
        """
        self.q_table[state + (action,)] += self.alpha * td_error

    def learn(self, num_episodes):
        """
            Description: This function implements the Q-Learning Procedure.
            Args:
                num_episodes : The number of episodes.
            Returns:
                reward_list  : The list of rewards.
                total_steps_list : The list of steps taken.
        """
        reward_list = []
        total_steps_list = []
        for episode in range(num_episodes):
            # initialize the environment
            self.env.reset(seed=self.seed)

            terminated = False # When reached the goal green square or collided with an obstacle
            truncated = False # When max_steps have been done

            # initialize the reward
            total_reward = 0
            tau = self.tau

            if episode % 100 == 0:
                # Decaying tau
                tau = tau * self.tau_decay
            
            steps = 0


            while not (terminated or truncated):
                steps += 1

                position = (self.env.agent_pos[0] - 1) * 3 + (self.env.agent_pos[1] - 1) # 0 to 9
                direction = self.env.agent_dir
                front_cell = self.env.grid.get(*self.env.front_pos)
                not_clear = front_cell and front_cell.type != 'goal'
                if not_clear:
                    immediate_cell = 1
                else:
                    immediate_cell = 0
                
                # get the action
                action = self.policy.get_action(tau, (position, direction, immediate_cell))
                # take the action
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                next_position = (self.env.agent_pos[0] - 1) * 3 + (self.env.agent_pos[1] - 1) # 0 to 9
                next_direction = self.env.agent_dir
                next_front_cell = self.env.grid.get(*self.env.front_pos)
                next_not_clear = next_front_cell and next_front_cell.type != 'goal'
                if not_clear:
                    next_immediate_cell = 1
                else:
                    next_immediate_cell = 0

                # print("next_state = ", next_state, type(next_state))
               # update the q-table
                td_error = self.compute_td_error((position, direction, immediate_cell), action, (next_position, next_direction, next_immediate_cell), reward)
                self.update_q_table((position, direction, immediate_cell), action, td_error)

                # update the total reward
                total_reward += reward
                # check if the episode is finished
                if terminated or truncated:
                    # print the total reward
                    print("Episode: {}/{}, Total Reward: {}, Total steps{}".format(episode + 1, num_episodes, total_reward, steps))
                    # append the total reward
                    reward_list.append(total_reward)
                    total_steps_list.append(steps)
                    break

        return reward_list, total_steps_list