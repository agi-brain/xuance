import argparse
import numpy as np
import torch
import torch.nn as nn
import time
import pickle
import random
import scipy.misc
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


class Buffer:
    def __init__(self, arglist, obs_dim, num_others):
        self.arglist = arglist
        self.obs_dim = obs_dim
        self.num_others = num_others


def mlp_model(input_dim, output_dim, num_layers=3, num_units=128):
    layers = []
    layers.append(nn.Linear(input_dim, num_units))
    layers.append(nn.ReLU())
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(num_units, num_units))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(num_units, output_dim))
    return nn.Sequential(*layers)


class MModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, num_units=128):
        super(MModel, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.lstm = nn.LSTM(input_dim, num_units, num_layers)
        self.fc = nn.Linear(num_units, output_dim)

    def forward(self, input):
        output, (h_n, c_n) = self.lstm(input)
        out = self.fc(output[:, -1, :])
        return out
def get_message(obs_n, target_pos_idx_n, num_agents_obs):
    num_agents = len(obs_n)
    obs_dim = obs_n[0].shape[-1]
    # if there exists no communication with other agents, message matrix will be zeros
    # messgae [zeros,zeros,o3,o4,zeros] for agent 1 when it communicates with 3,4
    message_n = [ np.zeros((num_agents_obs, obs_dim), dtype=np.float32) for _ in range(num_agents)]
    real_pos_n = []
    for j in range(num_agents):
        for jj in range(len(target_pos_idx_n[j])):
            message_n[j][jj,:] = obs_n[target_pos_idx_n[j][jj]]
    return message_n
def get_comm_pairs(obs_n, num_agents_obs, num_others):
    # 获取通信对的位置信息 other_loc_n 和索引信息 other_idx_n。
    # 通信对是指智能体之间需要进行通信的对，位置信息表示了需要进行通信的智能体相对于当前智能体的位置，
    # 索引信息表示了需要进行通信的智能体在环境中的索引。
    num_agents = len(obs_n)
    obs_dim = obs_n[0].shape[-1]
    target_loc_n = []
    target_idx_n = []
    target_idx = None
    real_loc_n = []
    # get the positions of each agent, the first two elements are the vel of each agent
    for i in range(num_agents):
        real_loc_n.append(obs_n[i][2:4])
    for i in range(num_agents):
        # remove the real_position and vel of agent, keep the relative position
        obs_tmp = obs_n[i][4:].copy()
        obs_tmp[0::2] = obs_tmp[0::2]+real_loc_n[i][0]
        obs_tmp[1::2] = obs_tmp[1::2]+real_loc_n[i][1]
        target_loc_all = []
        target_idx_all = []
        for j in range(num_agents_obs):
            target_loc = obs_tmp[int((num_others+j)*2): int((num_others+j)*2+2)]
            for ii in range(len(real_loc_n)):
                if (abs(real_loc_n[ii][0]-target_loc[0])<1e-5) and (abs(real_loc_n[ii][1]-target_loc[1])<1e-5):
                    target_idx = ii
            #tar_pos_all.append(obs_n[i][int((num_landmark+j)*2+4): int((num_landmark+j)*2+2+4)])
            target_loc_all.append(real_loc_n[i]-target_loc)
            target_idx_all.append(target_idx)
        target_loc_n.append(target_loc_all)
        target_idx_n.append(target_idx_all)
    return target_loc_n, target_idx_n

def make_env(arglist):
    scenario_name = arglist.scenario
    benchmark = arglist.benchmark
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env
def m_model(input, num_outputs, num_other, scope, num_layers=2, reuse=False, num_units=128, rnn_cell=None):
    hidden_size = num_units
    timestep_size = num_other
    with tf.variable_scope(scope, reuse=reuse):
        mlstm_cell=[]
        for _ in range(num_layers):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
            mlstm_cell.append(lstm_cell)
        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(mlstm_cell, state_is_tuple=True)
        outputs, state = tf.nn.dynamic_rnn(cell=mlstm_cell, inputs=input, dtype=tf.float32)
        out = outputs[:, -1, :]
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out
def get_trainers(arglist, env, num_adversaries, obs_shape_n, message_shape_n,target_loc_space_n, prior_buffer):
    trainers = []
    model = [mlp_model, m_model]
    trainer = AgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent", model, obs_shape_n, message_shape_n, target_loc_space_n, env.action_space, i, env.n_agents_obs, arglist, prior_buffer,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n_agents):
        trainers.append(trainer(
            "agent", model, obs_shape_n, message_shape_n, target_loc_space_n, env.action_space, i, env.n_agents_obs, arglist, prior_buffer,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def train(arglist):
    env = make_env(arglist)
    obs_n = env.reset()
    num_others = env.n_landmarks_obs if arglist.scenario == 'cn' else env.n_preys_obs
    other_loc_n, other_idx_n = get_comm_pairs(obs_n, env.n_agents_obs, num_others)

    obs_shape_n = [env.observation_space[0].shape for _ in range(env.n_agents)]
    message_shape_n = [(env.n_agents_obs,) + env.observation_space[0].shape for _ in range(env.n_agents)]
    target_loc_space_n = [(len(other_loc_n[0][0]),) for _ in range(env.n_agents)]
    num_adversaries = min(env.n_agents, arglist.num_adversaries)
    prior_buffer = Buffer(arglist, env.observation_space[0].shape[0], len(other_loc_n[0][0]))
    trainers = get_trainers(arglist, env, num_adversaries, obs_shape_n, message_shape_n, target_loc_space_n,
                            prior_buffer)

    episode_rewards = [0.0]
    comm_freq = [0.0]
    agent_rewards = [[0.0] for _ in range(env.n_agents)]
    final_ep_rewards = []
    final_ep_ag_rewards = []
    agent_info = [[[]]]
    episode_step = 0
    training_step = 0
    t_start = time.time()
    max_mean_epi_reward = -100000
    num_comm = 0
    print('Starting iterations...')

    while True:
        action_n = []
        target_idx_n = []
        for i, agent in enumerate(trainers):
            other_loc = other_loc_n[i]
            other_idx = other_idx_n[i]
            target_idx = []
            for j in range(len(other_loc)):
                if agent.target_comm(obs_n[i], np.array(other_loc[j])):
                    target_idx.append(other_idx[j])
            num_comm += len(target_idx)
            target_idx_n.append(target_idx)

        message_n = get_message(obs_n, target_idx_n, env.n_agents_obs)

        for i, agent in enumerate(trainers):
            action_n.append(agent.action(obs_n[i], message_n[i]))

        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        new_other_loc_n, new_other_idx_n = get_comm_pairs(new_obs_n, env.n_agents_obs, num_others)
        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew / len(rew_n)
            agent_rewards[i][-1] += rew

        if done or terminal:
            comm_freq[-1] = comm_freq[-1] / (num_others * env.n_agents * arglist.max_episode_len)
            episode_rewards[-1] = episode_rewards[-1] / arglist.max_episode_len
            obs_n = env.reset()
            other_loc_n, other_idx_n = get_comm_pairs(obs_n, env.n_agents_obs, num_others)
            episode_step = 0
            comm_freq.append(0.0)
            episode_rewards.append(0.0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

        training_step += 1

        loss = None
        for agent in trainers:
            agent.preupdate()
        training_idx = random.randint(0, env.n_agents - 1)
        loss = trainers[training_idx].update(trainers, training_step)

        prior_training_flag = True if (terminal and len(episode_rewards) % arglist.prior_training_rate == 0) else False
        if prior_training_flag:
            print("gathering prior training data...")
            is_full = trainers[training_idx].get_samples(trainers)
            if is_full:
                print("training prior network...")
                for _ in range(arglist.prior_num_iter):
                    trainers[training_idx].prior


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
