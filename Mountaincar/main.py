
import argparse
import gymnasium as gym
import numpy as np
import wandb.filesync
from q_table import Qtable
from sarsa_agent import SarasLearner
from q_agent import QLearner
from utils import plot_reward
import wandb
wandb.login()

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='q_learning', help='The algorithm to use. It can be either q_learning or sarsa.')
parser.add_argument('--alpha', type=float, default=0.1, help='The learning rate.')
parser.add_argument('--gamma', type=float, default=0.99, help='The discount factor.')
parser.add_argument('--epsilon', type=float, default=0.1, help='The exploration rate.')
parser.add_argument('--epsilon_decay', type=float, default=1.0, help='The exploration rate decay.')
parser.add_argument('--tau', type=float, default=1.0, help='The tau for softmax.')
parser.add_argument('--tau_decay', type=float, default=1.0, help='The tau_decay for softmax.')
parser.add_argument('--num_episodes', type=int, default=10000, help='The number of episodes.')
parser.add_argument('--num_steps', type=int, default=200, help='The number of steps.')
parser.add_argument('--num_bins', type=int, default=20, help='The number of bins for discretizing the state space.')
parser.add_argument('--seed', default=[10, 20, 30, 40, 50], help='The seed for the random number generator.')
parser.add_argument('--render', type = bool, default=False, help='To render or not to render the environment')
parser.add_argument('--wandb_project_name', type=str, required=True, help='The name to be given to the wandb project')
args = parser.parse_args()

# define the environment
# The minimizing regret part can be done by changing the reward function, by making it dependent on position, and velocity and combination of them. 
# However, i have taken the reward from the gym environment as it is, to make the algorithms run.
# Epsilon decay is done after every 100 episodes
# The reward function can be changed to minimize the regret but it can add to the complexity of interpretation of the plots of episode return vs episode. 
# So, we have stayed with the simpler reward as mentioned in the gym documentation. (Since, solving the mountaincar is a toy environment, we stayed with the simpler reward function only)
if args.render:
    env = gym.make('MountainCar-v0', render_mode='human')
else:
    env = gym.make('MountainCar-v0')

for seed in args.seed:
    np.random.seed(seed)
    print("seed = ", seed)
    reward_list_seed = []

    if args.algorithm == "q_learning":
        file_name = args.algorithm + "_alpha_" + str(args.alpha) +\
            "_gamma_" + str(args.gamma) + "_tau_" +\
            str(args.tau) + "_num_episodes_" +\
            str(args.num_episodes) + "_num_steps_" +\
            str(args.num_steps) + "_num_bins_" + str(args.num_bins)

        wandb.init(project=args.wandb_project_name, name=file_name + "_seed_" + str(seed))

        # define the q-table
        q_table, bins = Qtable(env.observation_space, env.action_space, args.num_bins)

        # instantiate the q-learner agent
        q_learner = QLearner(args.alpha, args.gamma, args.tau, q_table, bins, env, seed, args.tau_decay)
        # learn the q-table
        reward_list = q_learner.learn(args.num_episodes, args.num_steps)

        for reward in reward_list:
            wandb.log({"Episode_return": reward})
        
        wandb.log({'Max episode return': max(reward_list)})
        wandb.log({'Min episode return': min(reward_list)})
        wandb.finish()


    elif args.algorithm == "sarsa":
        file_name = args.algorithm + "_alpha_" + str(args.alpha) +\
            "_gamma_" + str(args.gamma) + "_epsilon_" +\
            str(args.epsilon) + "_num_episodes_" +\
            str(args.num_episodes) + "_num_steps_" +\
            str(args.num_steps) + "_num_bins_" + str(args.num_bins)

        wandb.init(project=args.wandb_project_name, name=file_name + "_seed_" + str(seed))
        # define the q-table
        q_table, bins = Qtable(env.observation_space, env.action_space, args.num_bins)

        # instantiate the q-learner agent
        sarsa_learner = SarasLearner(args.alpha, args.gamma, args.epsilon, q_table, bins, env, seed, args.epsilon_decay)
        # learn the q-table
        reward_list = sarsa_learner.learn(args.num_episodes, args.num_steps)

        for reward in reward_list:
            wandb.log({"Episode_return": reward})
        
        wandb.log({'Max episode return': max(reward_list)})
        wandb.log({'Min episode return': min(reward_list)})

        wandb.finish()

    else:
        raise ValueError("The algorithm should be either q_learning or sarsa.")

    reward_list_seed.append(reward_list)

# Plot
plot_reward(reward_list_seed=reward_list_seed,
            num_episodes=args.num_episodes,
            file_name=file_name + '.png')
