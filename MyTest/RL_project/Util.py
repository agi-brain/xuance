from collections import defaultdict

import numpy as np

def sarsa(env,obs,gamma,l_rate,eps,Q):
    # 在状态s下根据策略选择动作
    a = eps_greedy(env.action_space, eps, Q, obs)

    next_obs, reward, terminated, truncated, info = env.step(a)
    next_obs = tuple(next_obs[0].astype(int))
    next_a = eps_greedy(env.action_space, eps, Q, next_obs)

    # SARSA更新规则
    TD_target = reward + gamma * Q[next_obs, next_a]
    TD_error = TD_target - Q[obs, a]
    Q[obs, a] += l_rate * TD_error
    return next_obs,next_a


def expected_sarsa(env, obs, gamma, l_rate, eps, Q):
    # 在状态s下根据策略选择动作
    a = eps_greedy(env.action_space, eps, Q, obs)

    next_obs, reward, terminated, truncated, info = env.step(a)
    next_obs = tuple(next_obs[0].astype(int))
    next_a = eps_greedy(env.action_space, eps, Q, next_obs)
    # 计算期望Q值
    expected_q = 0.0
    q_values = [Q[(next_obs, action)] for action in range(env.action_space.n)]
    max_action = np.argmax(q_values)

    for action in range(env.action_space.n):
        if action == max_action:
            expected_q += (1 - eps + eps / env.action_space.n) * Q[(next_obs, action)]
        else:
            expected_q += (eps / env.action_space.n) * Q[(next_obs, action)]
    # 期望SARSA更新规则
    TD_target = reward + gamma * expected_q
    TD_error = TD_target - Q[obs, a]
    Q[obs, a] += l_rate * TD_error
    return next_obs,next_a

def q_learning(env,obs,gamma,l_rate,eps,Q):
    # 在状态s下根据策略选择动作
    a = eps_greedy(env.action_space, eps, Q, obs)

    next_obs, reward, terminated, truncated, info = env.step(a)
    next_obs = tuple(next_obs[0].astype(int))
    next_a = eps_greedy(env.action_space, eps, Q, next_obs)

    # q-learning更新规则
    TD_target = reward + gamma * max([Q[(next_obs, action)] for action in range(env.action_space.n)])
    TD_error = TD_target - Q[obs, a]
    Q[obs, a] += l_rate * TD_error
    return next_obs, next_a
def eps_greedy(action_space,epsilon,Q,obs)->int:#返回一个动作
    if np.random.rand() < epsilon:
        return action_space.sample()
    else:
        #如果动作为0，则在+1~+10中取所有后选状态的平均值;动作为1则取当前状态的值
        q_values = [Q[(tuple(obs), action)] for action in range(action_space.n)]
        return np.argmax(q_values)
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