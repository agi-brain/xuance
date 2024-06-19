from xuance.tensorflow.agents import *


class VDAC_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv,
                 device: str = "cpu:0"):
        self.gamma = config.gamma
        self.n_envs = envs.num_envs
        self.n_size = config.n_size
        self.n_epochs = config.n_epochs
        self.n_minibatch = config.n_minibatch
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        self.use_rnn = config.use_rnn
        # create representation for actor
        kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                      "dropout": config.dropout,
                      "rnn": config.rnn} if self.use_rnn else {}
        representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
        # create policy
        if config.mixer == "VDN":
            mixer = VDN_mixer()
        elif config.mixer == "QMIX":
            mixer = QMIX_mixer(config.dim_state[0], config.hidden_dim_mixing_net, config.hidden_dim_hyper_net,
                               config.n_agents, device)
        elif config.mixer == "Independent":
            mixer = None
        else:
            raise f"Mixer named {config.mixer} is not defined!"
        input_policy = get_policy_in_marl(config, representation, mixer=mixer)
        policy = REGISTRY_Policy[config.policy](*input_policy,
                                                use_rnn=config.use_rnn,
                                                rnn=config.rnn,
                                                gain=config.gain)
        lr_scheduler = MyLinearLR(config.learning_rate, start_factor=1.0, end_factor=0.5,
                                  total_iters=get_total_iters(config.agent_name, config))
        optimizer = tk.optimizers.Adam(lr_scheduler)
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.auxiliary_info_shape = {}

        buffer = MARL_OnPolicyBuffer_RNN if self.use_rnn else MARL_OnPolicyBuffer
        input_buffer = (config.n_agents, config.state_space.shape, config.obs_shape, config.act_shape, config.rew_shape,
                        config.done_shape, envs.num_envs, config.n_size,
                        config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda)
        memory = buffer(*input_buffer, max_episode_steps=envs.max_episode_steps, dim_act=config.dim_act)
        self.buffer_size = memory.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch

        learner = VDAC_Learner(config, policy, optimizer, config.device, config.model_dir, config.gamma)
        super(VDAC_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                          config.log_dir, config.model_dir)
        self.share_values = True if config.rew_shape[0] == 1 else False
        self.on_policy = True

    def act(self, obs_n, *rnn_hidden, avail_actions=None, state=None, test_mode=False):
        batch_size = len(obs_n)
        agents_id = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))
        obs_in = tf.reshape(tf.convert_to_tensor(obs_n), [batch_size, self.n_agents, -1])
        if state is not None:
            state = tf.convert_to_tensor(state)
        if self.use_rnn:
            batch_agents = batch_size * self.n_agents
            hidden_state, dists, values_tot = self.policy(obs_in.view(batch_agents, 1, -1),
                                                          agents_id.unsqueeze(2),
                                                          *rnn_hidden,
                                                          avail_actions=avail_actions[:, :, np.newaxis],
                                                          state=state.unsqueeze(2))
            actions = dists.stochastic_sample()
            actions = actions.reshape(batch_size, self.n_agents)
            values_tot = values_tot.reshape([batch_size, self.n_agents, 1])
        else:
            inputs = {'obs': obs_in, 'ids': agents_id}
            hidden_state, dists, values_tot = self.policy(inputs,
                                                          avail_actions=avail_actions,
                                                          state=state)
            actions = dists.stochastic_sample()
            values_tot = tf.reshape(values_tot, [batch_size, self.n_agents, 1])
        return hidden_state, actions.numpy(), values_tot.numpy()

    def train(self, i_step, **kwargs):
        if self.memory.full:
            info_train = {}
            indexes = np.arange(self.buffer_size)
            for _ in range(self.n_epochs):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    if self.use_rnn:
                        info_train = self.learner.update_recurrent(sample)
                    else:
                        info_train = self.learner.update(sample)
            self.learner.lr_decay(i_step)
            self.memory.clear()
            return info_train
        else:
            return {}
