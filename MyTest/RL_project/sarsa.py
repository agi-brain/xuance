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

def sample(env, timestep_max, number,gamma,V_table):
    ''' 采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number '''
    episodes = []
    for i in tqdm(range(number)):
        episode = []
        timestep = 0
        obs,info=env.reset()
        # 当前状态为终止状态或者时间步太长时,一次采样结束
        obs=obs.astype(int)
        terminated, truncated = False, False

        # 在状态s下根据策略选择动作
        while not terminated and timestep <= timestep_max:
            a = eps_greedy(env.action_space, 0.99*(number-i)/number, V_table,obs)
            next_obs, reward, terminated, truncated, info = env.step(a)
            next_obs = next_obs.astype(int)
            V_table[obs[0][0]][obs[0][1]][obs[0][2]] +=gamma**timestep * reward
            episode.append([obs, a, reward, next_obs])  # 把(obs, a, reward, next_obs)元组放入序列中
            obs = next_obs  # s_next变成当前状态,开始接下来的循环
            timestep += 1
        episodes.append(episode)
    return episodes


def MC(episodes, V_table, gamma,l_rate):
    for episode in tqdm(episodes):
        G = 0
        for i in range(len(episode) - 1, -1, -1):  #一个序列从后往前计算
            [obs, a, r, obs_next] = episode[i]
            obs = obs.astype(int)
            index_value=V_table[obs[0][0]][obs[0][1]][obs[0][2]]
            G = r + gamma * G
            V_table[obs[0][0]][obs[0][1]][obs[0][2]] = index_value+ (G - index_value) * l_rate

    np.save('V_table.npy', V_table)
def eps_greedy(action_space,epsilon,V_table,obs)->int:#返回一个动作
    obs=obs[0].astype(int)
    action_0,action_1=0,0
    if np.random.rand() < epsilon:
        return action_space.sample()
    else:
        #如果动作为0，则在+1~+10中取所有后选状态的平均值;动作为1则取当前状态的值
        for action in range(action_space.n):
            if action==0:#代表的意思是要所有的（在2-10以内）无A和有A，两类加起来平均
                action_0+=np.sum(V_table[obs[0]][obs[1]+2:obs[1]+10][obs[2]])+V_table[obs[0]][obs[1]][0 if obs[2]==1 else 1]
                non_zero_elements = V_table[obs[0]][obs[1] + 2:obs[1] + 10][0] != 0
                # 计算非零元素的数量
                count_non_zero = np.count_nonzero(non_zero_elements)
                if count_non_zero!=0:
                    action_0/=count_non_zero

            elif action==1:
                action_1=V_table[obs[0]][obs[1]][obs[2]]
        if action_0> action_1:
            return 0
        else:
            return 1

def MC_test(V_table,env,number):
    obs,info = env.reset()
    obs = obs.astype(int)

    res=[]
    episodes=[]
    for i in range(number):
        r=0
        episode=[]
        terminated, truncated = False, False
        while not terminated :
            a = eps_greedy(env.action_space, 0, V_table, obs)
            next_obs, reward, terminated, truncated, info = env.step(a)
            next_obs = next_obs.astype(int)
            episode.append([obs,next_obs])
            obs = next_obs  # s_next变成当前状态,开始接下来的循环
            r+=reward
        res.append(r)
        episodes.append(episode)
        obs,info=env.reset()
    return res,episodes

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
    V_table=np.zeros((12,30,2)) #一个12*30*2的表格来存储状态价值
    if not  Test:
        episodes = sample(env, 20, 500000, gamma,V_table)
        MC(episodes, V_table, gamma,l_rate=l_rate)
    else:
        V_table=np.load('V_table.npy')
        res,ep=MC_test(V_table,env,50000)
        win=0
        fail=0
        for i in res:
            if i>0:
                win+=1
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
    # V_table = np.zeros((12, 30, 2))  # 一个12*30*2的表格来存储状态价值
    # V_table=np.load('V_table.npy')
    # print(V_table)
