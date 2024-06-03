from collections import defaultdict

from BlackjackEnv import BlackjackEnv
import numpy as np
import argparse
from agent import DQN_Agent
from tqdm import tqdm
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

def save_Q(Q, filename):
    # 将字典转换为列表
    Q_list = [(k, v) for k, v in Q.items()]
    # 保存为npy文件
    np.save(filename, Q_list)

def load_Q(filename):
    # 加载npy文件
    Q_list = np.load(filename, allow_pickle=True)
    # 将列表转换为字典
    Q = defaultdict(lambda: 0.0)
    for k, v in Q_list:
        Q[k] = v
    return Q
def sample(env, timestep_max, number, Q, l_rate, gamma):
    ''' 采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number '''
    for i in tqdm(range(number)):
        obs, info = env.reset()
        obs = tuple(obs[0].astype(int))
        terminated, truncated = False, False
        timestep = 0
        # 在状态s下根据策略选择动作
        while not terminated and timestep < timestep_max:
            # 选择初始动作
            a = eps_greedy(env.action_space, 0.99 * (number - i) / number, Q, obs)
            next_obs, reward, terminated, truncated, info = env.step(a)
            next_obs = tuple(next_obs[0].astype(int))
            # q-learning更新规则
            TD_target = reward + gamma * max([Q[(next_obs, action)] for action in range(env.action_space.n)] )
            TD_error = TD_target - Q[obs, a]
            Q[obs,a] += l_rate * TD_error
            obs = next_obs
            timestep += 1
    return  Q


def eps_greedy(action_space,epsilon,Q,obs)->int:#返回一个动作
    if np.random.rand() < epsilon:
        return action_space.sample()
    else:
        #如果动作为0，则在+1~+10中取所有后选状态的平均值;动作为1则取当前状态的值
        q_values = [Q[(tuple(obs), action)] for action in range(action_space.n)]
        return np.argmax(q_values)

def q_learning_test(Q,env,number):
    obs,info = env.reset()
    res=[]
    for i in range(number):
        r=0
        obs = tuple(obs[0].astype(int))
        terminated, truncated = False, False
        while not terminated :
            a = eps_greedy(env.action_space, 0, Q, obs)
            next_obs, reward, terminated, truncated, info = env.step(a)
            next_obs = tuple(next_obs[0].astype(int))
            obs = next_obs  # s_next变成当前状态,开始接下来的循环
            r+=reward
        res.append(r)
        obs,info=env.reset()
    return res

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
    gamma = 0.99
    l_rate=0.001
    Test=True
    Q = defaultdict(lambda: 0.0) #一个map来存储Q值，key=(state,action), value=Q值
    if not  Test:
        Q=sample(env,20, 500000,Q, l_rate, gamma)
        save_Q(Q, 'Q.npy')
        # q_learning(episodes, Q, gamma,l_rate=l_rate)
    else:
        Q=load_Q('Q.npy')
        res=q_learning_test(Q,env,50000)
        win=0
        fail=0
        for i in res:
            if i>0:
                win+=1
            elif i==0:
                pass
            else:
                fail+=1
        print("win:",win,"fail:",fail)


if __name__ == '__main__':
    parser = parse_args()
    args = get_arguments(method=parser.method,
                         env=parser.env,
                         env_id=parser.env_id,
                         config_path=parser.config,
                         parser_args=parser)
    run(args)

