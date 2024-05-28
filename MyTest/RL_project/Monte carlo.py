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
    S = env.observation_space  # 状态集合
    A = env.action_space  # 动作集合

    gamma = 1.0  # 没有折扣
    MDP = (S, A, gamma)

# 把输入的两个字符串通过“-”连接,便于使用上述定义的P、R变量
def join(str1, str2):
    return str1 + '-' + str2


if __name__ == '__main__':
    parser = parse_args()
    args = get_arguments(method=parser.method,
                         env=parser.env,
                         env_id=parser.env_id,
                         config_path=parser.config,
                         parser_args=parser)
    run(args)

def sample(MDP,env, Pi, timestep_max, number):
    ''' 采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number '''
    S, A, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        obs,info=env.reset()
        # 当前状态为终止状态或者时间步太长时,一次采样结束
        terminated, truncated = False, False
        Q={0:0,1:0} #0和1分别代表动作twist和stick
        while terminated or timestep <= timestep_max:


            # 在状态s下根据策略选择动作
            for a in A:
                next_obs, reward, terminated, truncated, info = env.step(a)
                Q[a] +=gamma**timestep * reward

            # episode.append((s, a, r, s_next))  # 把（s,a,r,s_next）元组放入序列中
            # s = s_next  # s_next变成当前状态,开始接下来的循环
            timestep += 1
        episodes.append(episode)
    return episodes



def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode) - 1, -1, -1):  #一个序列从后往前计算
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]

def eps_greedy(action_space,epsilon,Q):#返回一个动作
    if np.random.rand() < epsilon:
        return action_space.sample()
    else:
        return np.argmax(Q[action_space])

