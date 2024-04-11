import numpy as np
import random
import tensorflow as tf
import I2C.common.tf_util as U
from I2C.common.distributions import make_pdtype
from I2C import AgentTrainer
from I2C.trainer.replay_buffer import ReplayBuffer
import math

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_m_train(make_obs_ph_n, make_message_ph_n, act_space_n, num_agents_obs, p_index, m_func, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=128, scope="trainer", reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        message_ph_n = make_message_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        blz_distribution = tf.placeholder(tf.float32, [None, act_space_n[p_index].n], name="blz_distribution")
        m_input = message_ph_n[p_index]
        encode_dim = m_input.get_shape().as_list()[-1]
        # message encoder
        message_encode = m_func(m_input,encode_dim, num_agents_obs, scope='m_func', num_units=num_units)
        m_func_vars = U.scope_vars(U.absolute_scope_name("m_func"))
        # policy
        p_input = tf.concat((obs_ph_n[p_index], message_encode), 1)   
        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", type='fit', num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)
        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
        act_input_n = act_ph_n + []
        # correlation reg
        k = tf.keras.losses.KLDivergence() 
        KL_reg = k(blz_distribution, act_sample)
        # q network
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", type='fit', reuse=True, num_units=num_units)[:,0]
        # loss and optimization
        pg_loss = -tf.reduce_mean(q)
        loss = pg_loss + KL_reg * 1e-2
        optimize_expr = U.minimize_and_clip(optimizer, loss, [p_func_vars,m_func_vars], grad_norm_clipping)
        # Create callable functions
        train = U.function(inputs=obs_ph_n + message_ph_n + act_ph_n+[blz_distribution], outputs= loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index], message_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index], message_ph_n[p_index]], outputs=p)
        # target network
        target_message_encode = m_func(m_input, encode_dim, num_agents_obs, scope='target_m_func', num_units=num_units)
        target_m_func_vars = U.scope_vars(U.absolute_scope_name("target_m_func"))
        p_input = tf.concat((obs_ph_n[p_index], target_message_encode), 1)
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func",type='fit', num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_m = make_update_exp(m_func_vars, target_m_func_vars)
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)
        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index], message_ph_n[p_index]], outputs=target_act_sample)
        return act, train, update_target_p, update_target_m, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=tf.AUTO_REUSE, num_units=128):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        # q network
        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", type='fit', num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        q_loss = tf.reduce_mean(tf.square(q - target_ph))
        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg
        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)
        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)
        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", type='fit', num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)
        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)
        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

def c_train(make_obs_ph_n, make_target_loc_ph_n, c_index, c_func, q_func, optimizer, scope="trainer", num_units=128, grad_norm_clipping=None , reuse=tf.AUTO_REUSE, local_q_func = False):
    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        target_loc_ph = make_target_loc_ph_n[c_index]#tf.placeholder(tf.float32, [None,2], name="target_loc")
        self_obs_ph = obs_ph_n[c_index]
        labels_ph = tf.placeholder(tf.float32, [None,2], name="labels")
        # prior network
        c_input = tf.concat((self_obs_ph,target_loc_ph), 1)
        c = c_func(c_input, 2, scope="c_func", type='cls', num_units=num_units)
        c_pred = tf.nn.softmax(c)
        c_flags = tf.greater(c_pred[:,0],0.5)
        c_func_vars = U.scope_vars(U.absolute_scope_name("c_func"))
        # loss and optimization
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=c, labels=labels_ph))
        optimize_expr = U.minimize_and_clip(optimizer, loss, c_func_vars, grad_norm_clipping)
        # Create callable functions
        c_train = U.function(inputs=[obs_ph_n[c_index], target_loc_ph, labels_ph], outputs=loss, updates=[optimize_expr])
        c_act = U.function(inputs=[obs_ph_n[c_index], target_loc_ph], outputs=c_flags)
        c_values = U.function([obs_ph_n[c_index], target_loc_ph], outputs = c_pred)
        # target network
        target_c_values = c_func(c_input, 2, scope="target_c_func", type='cls', num_units=num_units)
        target_c_pred = tf.nn.softmax(target_c_values)
        target_c_flags = tf.greater(target_c_pred[:,0],0.5)
        target_c_func_vars = U.scope_vars(U.absolute_scope_name("target_c_func"))
        update_target_c = make_update_exp(c_func_vars, target_c_func_vars)
        target_c_act = U.function(inputs=[obs_ph_n[c_index], target_loc_ph], outputs=target_c_flags)
        return c_act, c_train, update_target_c,  {'c_values': c_values, 'target_c_act': target_c_act}


class AgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, message_shape_n, target_loc_space_n, act_space_n, agent_index, num_agents_obs, args, prior_buffer,local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        message_ph_n = []
        target_loc_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())
            message_ph_n.append(U.BatchInput(message_shape_n[i], name="message"+str(i)).get())
            target_loc_ph_n.append(U.BatchInput(target_loc_space_n[i], name="target_location"+str(i)).get())
        self.num_agents_obs = num_agents_obs
        self.model = model
        self.obs_ph_n = obs_ph_n
        self.message_ph_n = message_ph_n
        self.target_loc_ph_n = target_loc_ph_n
        self.local_q_func = local_q_func
        self.act_space_n = act_space_n  
        # Create experience buffer
        self.replay_buffer_general = ReplayBuffer(1e6)
        self.prior_buffer = prior_buffer
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        self.c_train = None
        self.c_act = None
        self.c_update = None
        self.c_debug = None
        self.act = None
        self.p_m_train = None
        self.p_update =None
        self.m_update=None
        self.p_m_debug=None
        self.q_train=None
        self.q_update=None
        self.q_debug=None
        self.step = 0
        
    def initial_c_model(self):
        self.c_act, self.c_train, self.c_update, self.c_debug = c_train(
            scope=self.name,
            make_obs_ph_n=self.obs_ph_n,
            make_target_loc_ph_n=self.target_loc_ph_n,
            c_index=self.agent_index,
            c_func=self.model[0],
            q_func=self.model[0],
            optimizer=tf.train.AdamOptimizer(learning_rate=self.args.lr),
            grad_norm_clipping=0.5,
            local_q_func=self.local_q_func,
            num_units=self.args.num_units
        )
        
    def initial_p_m_model(self):
        self.act, self.p_m_train, self.p_update, self.m_update, self.p_m_debug = p_m_train(
            scope=self.name,
            make_obs_ph_n=self.obs_ph_n,
            make_message_ph_n = self.message_ph_n,
            act_space_n= self.act_space_n,
            num_agents_obs= self.num_agents_obs,
            p_index= self.agent_index,
            m_func = self.model[1],
            p_func= self.model[0],
            q_func= self.model[0],
            optimizer=tf.train.AdamOptimizer(learning_rate=self.args.lr),
            grad_norm_clipping=0.5,
            local_q_func=self.local_q_func,
            num_units=self.args.num_units
        )

    def initial_q_model(self):
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=self.obs_ph_n,
            act_space_n=self.act_space_n,
            q_index=self.agent_index,
            q_func=self.model[0],
            optimizer=tf.train.AdamOptimizer(learning_rate=self.args.lr),
            grad_norm_clipping=0.5,
            local_q_func=self.local_q_func,
            num_units=self.args.num_units
        )
    def target_comm(self, obs, target_loc):
        return self.c_act(obs[None], target_loc[None])[0]

    def action(self, obs, message):
        return self.act(obs[None], message[None])[0]

    def q_value(self, obs_n, act_n):
        return self.q_debug['q_values'](*(obs_n+act_n))

    def experience(self, data):#obs, message, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer. 
        self.replay_buffer_general.add(data) #obs, message, act, rew, new_obs, float(done)

    def preupdate(self):
        self.replay_sample_index = None

    def blz_distribution(self, obs_n, act_n):
        action_space = self.act_space_n[self.agent_index].n
        q_values = []
        blz_distribution = []
        lambda_value = 8
        for i in range(action_space):
            one_hot = [0]*action_space
            one_hot[i] = 1
            act_n[self.agent_index][:,:] = one_hot[:]
            q_values.append(self.q_debug['q_values'](*(obs_n+act_n))[:,None]+1e-6)
        q_values = np.concatenate(q_values,1)
        # normalize
        q_values = q_values - np.mean(q_values,1)[:,None]
        q_values = q_values/np.max(q_values,1)[:,None]
        # calculate blz distribution
        q_values = np.exp(lambda_value*q_values)
        q_sum = np.sum(q_values,1)[:,None]
        blz_distribution = q_values/q_sum
        return blz_distribution

    def update(self, agents, t):
        if len(self.replay_buffer_general) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return 
        # collect replay sample from all agents
        batch_size = self.args.batch_size
        self.replay_sample_index = self.replay_buffer_general.make_index(batch_size) 
        obs_n = []
        obs_next_n = []
        act_n = []
        message_n = []
        target_loc_next_n = []
        target_idx_next_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, target_loc, target_idx, message, act, rew, obs_next, target_loc_next, target_idx_next, done = agents[i].replay_buffer_general.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            message_n.append(message)
            act_n.append(act)
            target_loc_next_n.append(target_loc_next)
            target_idx_next_n.append(target_idx_next)
        obs, target_loc, target_idx, message, act, rew, obs_next, target_loc_next, target_idx_next, done = self.replay_buffer_general.sample_index(index)
        # shape of target_loc_next (batch_size, num_agents_obs, obs_dim)
        num_agents_obs = self.num_agents_obs
        message_next_n = [np.zeros((batch_size, num_agents_obs, len(obs_next[0]))) for i in range(self.n)]
        # get message for next step
        flags_n_tmp = []
        for i in range(self.n):
            flags_tmp = []
            for j in range(num_agents_obs):
                flags_tmp.append(agents[i].c_debug['target_c_act'](*([obs_next_n[i]]+[target_loc_next_n[i][:,j,:]])))
            flags_n_tmp.append(flags_tmp)
        for i in range(batch_size):
            for j in range(self.n):
                for k in range(num_agents_obs):
                        target_idx = target_idx_next_n[j][i,k]
                        idx_tmp = 0
                        if flags_n_tmp[j][k][i] == True:
                            message_next_n[j][i, idx_tmp, :] = obs_next_n[target_idx][i,:]
                            idx_tmp = idx_tmp + 1
        # train q network
        num_sample = 1
        target_q = 0.0
        for j in range(num_sample):
            target_act_next_n = [agents[i].p_m_debug['target_act'](*([obs_next_n[i]]+[message_next_n[i]])) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))
        # train p and m network
        blz_distribution = self.blz_distribution(obs_n, act_n)
        p_loss = self.p_m_train(*(obs_n + message_n + act_n + [blz_distribution]))
        c_loss = None   
        # update p_m, q target network   
        self.p_update()
        self.m_update()
        self.q_update()
        return [q_loss, p_loss, c_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]

    def get_KL_value(self, obs_n, act_n, target_loc, target_idx):
        act_dim_self = len(act_n[self.agent_index][0])
        sample_size = len(obs_n[0])
        KL_values = []
        num_agents = len(obs_n)
        target_loc_input_n = [[] for _ in range(num_agents)]
        obs_act_idx = [[] for _ in range(num_agents)]
        for i in range(sample_size):
            for j in range(self.num_agents_obs):
                idx_tmp = target_idx[i,j]
                target_loc_input_n[idx_tmp].append(target_loc[i,j,:])
                obs_act_idx[idx_tmp].append(i)
        obs_input_n = [ [obs_n[n][obs_act_idx[k],:] for n in range(num_agents)] for k in range(num_agents)]
        act_input_n = [ [act_n[n][obs_act_idx[k],:] for n in range(num_agents)] for k in range(num_agents)]
        for i in range(num_agents):
            if i == self.agent_index or len(obs_act_idx[i])==0: continue
            act_dim_other = len(act_n[i][0])
            obs_input = obs_input_n[i][:]
            act_input = act_input_n[i][:]
            act_target = act_input[i][:,:].copy()
            Q_s = []
            Q_s_t = []
            for k in range(act_dim_self):
                one_hot = [0]*act_dim_self
                one_hot[k] = 1
                act_input[self.agent_index][:,:] = one_hot[:]
                Q_s.append(np.exp(self.q_debug['q_values'](*(obs_input+act_input)))+1e-8)
                Q_tmp = []
                for m in range(act_dim_other):
                    one_hot = [0]*act_dim_other
                    one_hot[m] = 1
                    act_input[i][:,:] = one_hot[:]
                    Q_tmp.append(np.exp(self.q_debug['q_values'](*(obs_input+act_input)))+1e-8)
                act_input[i][:,:] = act_target[:,:]
                Q_s_t.append(Q_tmp)    
            Q_t_sum = [sum(Q_s_t[ii]) for ii in range(act_dim_self)]
#            print(sum(Q_t_sum))
#            print(sum(Q_s)) 
            prob_s_marg = np.array(Q_t_sum/sum(Q_t_sum))
            prob_s_cond_t = np.array(Q_s/sum(Q_s))
            KL_value = np.sum(prob_s_marg*np.log(prob_s_marg / prob_s_cond_t),0)
            KL_values.append(KL_value)
        KL_values = np.concatenate(KL_values,0)
        obs_inputs = np.concatenate([obs_input_n[ii][self.agent_index] for ii in range(num_agents)],0) 
        while [] in target_loc_input_n:
            target_loc_input_n.remove([])
        target_loc_inputs = np.concatenate(target_loc_input_n, 0)
        return obs_inputs, target_loc_inputs, KL_values

    def prior_train(self, batch_size):
        obs_inputs, obs_loc_inputs, labels = self.prior_buffer.get_samples(batch_size)
        c_loss = self.c_train(*([obs_inputs, obs_loc_inputs, labels]))
        self.c_update()

    def get_samples(self, agents):
        self.replay_sample_index = self.replay_buffer_general.make_index(self.args.prior_buffer_size)
        obs_n = []
        obs_next_n = []
        act_n = []
        message_n = []
        target_loc_next_n = []
        target_idx_next_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, obs_loc, target_idx, message, act, rew, obs_next, target_loc_next, target_idx_next, done = agents[i].replay_buffer_general.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            message_n.append(message)
            act_n.append(act)
            target_loc_next_n.append(target_loc_next)
            target_idx_next_n.append(target_idx_next)
        obs, target_loc, target_idx, message, act, rew, obs_next, target_loc_next, target_loc_idx_next, done = self.replay_buffer_general.sample_index(index)
        obs_inputs, target_loc_inputs, KL_values = self.get_KL_value(obs_n, act_n, target_loc, target_idx)
        is_full = self.prior_buffer.insert(len(obs_inputs), obs_inputs, target_loc_inputs, KL_values)
        return is_full

