from BlackjackEnv import BlackjackEnv
import numpy as np
import argparse
from agent import DQN_Agent

import os
from copy import deepcopy
import numpy as np
import torch.optim
from xuance import get_arguments
from xuance.common import space2shape
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.utils import ActivationFunctions

def parse_args():
    parser = argparse.ArgumentParser("Run a demo.")
    parser.add_argument("--method", type=str, default="dqn")
    parser.add_argument("--env", type=str, default="BlackjackEnv")
    parser.add_argument("--env-id", type=str, default="blackjack-v0")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config", type=str, default="./configs/test_blackjack.yaml")
    return parser.parse_args()

def sample(env, timestep_max, number,gamma):
    ''' 采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number '''
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        obs,info=env.reset()
        # 当前状态为终止状态或者时间步太长时,一次采样结束
        terminated, truncated = False, False
        Q={0:0,1:0} #0和1分别代表动作twist和stick
        # 在状态s下根据策略选择动作
        while not terminated and timestep <= timestep_max:
            a = eps_greedy(env.action_space, 0.1, Q)
            next_obs, reward, terminated, truncated, info = env.step(a)
            Q[a] +=gamma**timestep * reward
            episode.append((obs, a, reward, next_obs))  # 把(obs, a, reward, next_obs)元组放入序列中
            obs = next_obs  # s_next变成当前状态,开始接下来的循环
            timestep += 1
        episodes.append(episode)
    return episodes


def MC(episodes, Q, gamma,l_rate):
    for episode in episodes:
        G = 0
        for i in range(len(episode) - 1, -1, -1):  #一个序列从后往前计算
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            Q[a] = Q[a] + (G - Q[a]) * l_rate

def eps_greedy(action_space,epsilon,Q):#返回一个动作
    if np.random.rand() < epsilon:
        return action_space.sample()
    else:
        # 获取所有可能的动作
        possible_actions = [action_space.sample() for _ in range(action_space.n)]
        best_action = np.argmax([Q[action] for action in possible_actions])
        return best_action


def run(args):
    agent_name = args.agent  # get the name of Agent.
    set_seed(args.seed)  # set random seed.

    # prepare directories for results
    args.model_dir = os.path.join(os.getcwd(), args.model_dir, args.env_id)  # the path for saved model.
    args.log_dir = os.path.join(args.log_dir, args.env_id)  # the path for logger file.

    # build environments
    env = make_envs(args)  # create simulation environments
    args.observation_space = env.observation_space  # get observation space
    args.action_space = env.action_space  # get action space
    gamma = 1.0  # 没有折扣
    # 采样5次,每个序列最长不超过10步
    episodes = sample(env, 10, 5, gamma)
    print('第一条序列\n', episodes[0][0])
    print('第二条序列\n', episodes[1][0])
    print('第五条序列\n', episodes[4][0])



if __name__ == '__main__':
    parser = parse_args()
    args = get_arguments(method=parser.method,
                         env=parser.env,
                         env_id=parser.env_id,
                         config_path=parser.config,
                         parser_args=parser)
    run(args)
