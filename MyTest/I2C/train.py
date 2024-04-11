import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
from tensorflow.contrib import rnn
import I2C.common.tf_util as U
from I2C.trainer.trainer import AgentTrainer
import tensorflow.contrib.layers as layers
import random
import scipy.misc
from I2C.trainer.storage import Buffer

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="cn", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=40, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=400000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=800, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--prior-batch-size", type=int, default=2000, help="number of samples to optimize at the same time for prior network")
    parser.add_argument("--prior-buffer-size", type=int, default=400000, help="prior network training buffer size")
    parser.add_argument("--prior-num-iter", type=int, default=10000, help="prior network training iterations")
    parser.add_argument("--prior-training-rate", type=int, default=20000, help="prior network training rate")
    parser.add_argument("--prior-training-percentile", type=int, default=80, help="control threshold for KL value to get labels")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='exp', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./tmp/policy/", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore_all", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

# message encoding network 
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

# prior network or action policy or Q function
def mlp_model(input, num_outputs, scope, type='fit', num_layer=3, reuse=False, num_units=128, rnn_cell=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        for i in range(num_layer-1):
            out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(arglist):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    scenario_name = arglist.scenario
    benchmark = arglist.benchmark
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

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

def train(arglist):
    with U.make_session():
        # Create environment
        env = make_env(arglist)
        obs_n = env.reset()
        num_others = env.n_landmarks_obs if arglist.scenario == 'cn' else env.n_preys_obs
        other_loc_n, other_idx_n = get_comm_pairs(obs_n, env.n_agents_obs, num_others)

        # Create agent trainers
        # Must have the same shape, otherwise cannot feed into NN
        obs_shape_n = [env.observation_space[0].shape for _ in range(env.n_agents)]
        message_shape_n = [ (env.n_agents_obs,)+env.observation_space[0].shape for _ in range(env.n_agents)]
        target_loc_space_n = [(len(other_loc_n[0][0]),) for _ in range(env.n_agents)]
        num_adversaries = min(env.n_agents, arglist.num_adversaries)
        prior_buffer = Buffer(arglist, env.observation_space[0].shape[0], len(other_loc_n[0][0]))
        trainers = get_trainers(arglist, env, num_adversaries, obs_shape_n, message_shape_n, target_loc_space_n, prior_buffer)
        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore_all:
            print('Loading I2C...')
            for i in range(env.n_agents):
                trainers[i].initial_q_model()
                trainers[i].initial_p_m_model()
                trainers[i].initial_c_model()
            U.initialize()
            U.load_state(arglist.load_dir)
        else:
            for i in range(env.n_agents):
                trainers[i].initial_q_model()
                trainers[i].initial_p_m_model()
                trainers[i].initial_c_model()
            U.initialize()

        episode_rewards = [0.0]  # sum of rewards for all agents
        comm_freq = [0.0]
        agent_rewards = [[0.0] for _ in range(env.n_agents)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        episode_step = 0
        training_step = 0
        t_start = time.time()
        max_mean_epi_reward = -100000
        num_comm = 0
        print('Starting iterations...')
        while True:
            # get messages
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
            # get actions
            for i, agent in enumerate(trainers):
                action_n.append(agent.action(obs_n[i], message_n[i]))
            # environment step    
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            new_other_loc_n, new_other_idx_n = get_comm_pairs(new_obs_n, env.n_agents_obs, num_others)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience([obs_n[i], other_loc_n[i], other_idx_n[i], message_n[i], action_n[i], rew_n[i], new_obs_n[i], new_other_loc_n[i], new_other_idx_n[i],done_n[i]])#, terminal))
            obs_n = new_obs_n
            other_loc_n = new_other_loc_n
            other_idx_n = new_other_idx_n
            # get episode reward and comm freq
            comm_freq[-1] += num_comm
            num_comm = 0
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew/len(rew_n)
                agent_rewards[i][-1] += rew
            # reset
            if done or terminal:
                comm_freq[-1] = comm_freq[-1]/(num_others*env.n_agents*arglist.max_episode_len)
                episode_rewards[-1]=episode_rewards[-1]/arglist.max_episode_len
                obs_n = env.reset()
                other_loc_n, other_idx_n = get_comm_pairs(obs_n, env.n_agents_obs, num_others)
                episode_step = 0
                comm_freq.append(0.0)
                episode_rewards.append(0.0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
            # increment global step counter
            training_step += 1
            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            training_idx = random.randint(0, env.n_agents-1)
            loss = trainers[training_idx].update(trainers, training_step)
            # sample data and training for prior network
            prior_training_flag = True if (terminal and len(episode_rewards) % arglist.prior_training_rate == 0) else False
            if prior_training_flag:
                print("gathering prior training data...")
                is_full = trainers[training_idx].get_samples(trainers)  
                if is_full:
                    print("training prior network...")
                    for _ in range(arglist.prior_num_iter):
                        trainers[training_idx].prior_train(arglist.prior_batch_size)
                    prior_training_flag = False

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                mean_epi_reward = np.mean(episode_rewards[-arglist.save_rate:])
                mean_comm_freq = np.mean(comm_freq[-arglist.save_rate:])
                if mean_epi_reward > max_mean_epi_reward:
                    U.save_state(arglist.save_dir, saver=saver)
                    max_mean_epi_reward = mean_epi_reward
                    print("save checkpoint...")   
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean comm freq: {}, mean episode reward: {}, time: {}".format(
                        training_step, len(episode_rewards), mean_comm_freq, mean_epi_reward, round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean comm freq: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        training_step, len(episode_rewards), mean_comm_freq, mean_epi_reward,
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
