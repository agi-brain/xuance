from collections import defaultdict

import argparse
from tqdm import tqdm
import os
from xuance import get_arguments
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed

from Util import save_Q,load_Q
from Util import sarsa,expected_sarsa,eps_greedy,q_learning
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

def sample(env, timestep_max, number, Q, l_rate, gamma,Alg):
    ''' 采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number '''
    for i in tqdm(range(number)):
        obs, info = env.reset()
        obs = tuple(obs[0].astype(int))
        terminated, truncated = False, False
        timestep = 0
        while not terminated and timestep < timestep_max:
            # 选择初始动作
            eps=0.99 * (number - i) / number
            if Alg=='sarsa':
                obs, a = sarsa(env, obs, gamma, l_rate, eps, Q)
            elif Alg=='expected_sarsa':
                obs, a = expected_sarsa(env,obs,gamma,l_rate,eps,Q)
            elif Alg=='q_learning':
                obs, a = q_learning(env, obs, gamma, l_rate, eps, Q)
            timestep += 1
    return Q

def Alg_test(Q,env,number):
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

def run(args,Alg,Test=False):
    agent_name = args.agent  # get the name of Agent.
    set_seed(args.seed)  # set random seed.

    # prepare directories for results
    args.model_dir = os.path.join(os.getcwd(), args.model_dir, args.env_id)  # the path for saved model.
    args.log_dir = os.path.join(args.log_dir, args.env_id)  # the path for logger file.

    # build environments
    env = make_envs(args)  # create simulation environments
    args.observation_space = env.observation_space  # get observation space
    args.action_space = env.action_space  # get action space
    Q = defaultdict(lambda: 0.0) #一个map来存储Q值，key=(state,action), value=Q值
    if not Test:
        Q=sample(env,20, 500000,Q, args.learning_rate, args.gamma,Alg)
        save_Q(Q, f'{Alg}.npy')
    else:
        Q=load_Q(f'{Alg}.npy')
        res=Alg_test(Q,env,50000)
        win=0
        fail=0
        for i in res:
            if i>0:
                win+=1
            elif i==0:
                pass
            else:
                fail+=1
        print(f"{Alg}测试结果：")
        print("win:",win,"fail:",fail)
        print("获胜率: %.2f" % (win / (win+fail)))
        print("失败率: %.2f" % (fail / (win + fail)))

if __name__ == '__main__':
    parser = parse_args()
    args = get_arguments(method=parser.method,
                         env=parser.env,
                         env_id=parser.env_id,
                         config_path=parser.config,
                         parser_args=parser)
    Alg='q_learning'# 选择算法，可选sarsa,expected_sarsa,q_learning
    run(args,Alg,Test=True)

