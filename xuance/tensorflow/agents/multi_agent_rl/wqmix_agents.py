from xuance.tensorflow.agents import *
from xuance.tensorflow.agents.agents_marl import linear_decay_or_increase


class WQMIX_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 device: str = "cpu:0"):
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy

        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        self.use_recurrent = config.use_recurrent
        if self.use_recurrent:
            kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                          "dropout": config.dropout,
                          "rnn": config.rnn}
            representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
        else:
            representation = REGISTRY_Representation[config.representation](*input_representation)
        mixer = QMIX_mixer(config.dim_state[0], config.hidden_dim_mixing_net, config.hidden_dim_hyper_net,
                           config.n_agents, device)
        ff_mixer = QMIX_FF_mixer(config.dim_state[0], config.hidden_dim_ff_mix_net, config.n_agents)
        input_policy = get_policy_in_marl(config, representation, mixer=mixer, ff_mixer=ff_mixer)
        policy = REGISTRY_Policy[config.policy](*input_policy,
                                                use_recurrent=config.use_recurrent,
                                                rnn=config.rnn)
        lr_scheduler = MyLinearLR(config.learning_rate, start_factor=1.0, end_factor=0.5,
                                  total_iters=get_total_iters(config.agent_name, config))
        optimizer = tk.optimizers.Adam(lr_scheduler)
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        buffer = MARL_OffPolicyBuffer_RNN if self.use_recurrent else MARL_OffPolicyBuffer
        input_buffer = (config.n_agents, state_shape, config.obs_shape, config.act_shape, config.rew_shape,
                        config.done_shape, envs.num_envs, config.buffer_size, config.batch_size)
        memory = buffer(*input_buffer, max_episode_length=envs.max_episode_length, dim_act=config.dim_act)

        learner = WQMIX_Learner(config, policy, optimizer,
                                config.device, config.model_dir, config.gamma, config.sync_frequency)
        super(WQMIX_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                           config.log_dir, config.model_dir)
        self.on_policy = False

    def act(self, obs_n, *rnn_hidden, avail_actions=None, test_mode=False):
        batch_size = obs_n.shape[0]
        agents_id = tf.repeat(tf.expand_dims(tf.eye(self.n_agents), 0), batch_size, 0)
        obs_in = tf.reshape(tf.convert_to_tensor(obs_n), [batch_size, self.n_agents, -1])
        if self.use_recurrent:
            batch_agents = batch_size * self.n_agents
            input_policy = {'obs': obs_in.view(batch_agents, 1, -1),
                            'ids': agents_id.view(batch_agents, 1, -1)}
            hidden_state, greedy_actions, _ = self.policy(input_policy,
                                                          *rnn_hidden,
                                                          avail_actions=avail_actions.reshape(batch_agents, 1, -1))
            greedy_actions = greedy_actions.view(batch_size, self.n_agents)
        else:
            input_policy = {'obs': obs_in, 'ids': agents_id}
            hidden_state, greedy_actions, _ = self.policy(input_policy, avail_actions=avail_actions)
        greedy_actions = greedy_actions.numpy()

        if test_mode:
            return hidden_state, greedy_actions
        else:
            if avail_actions is None:
                random_actions = np.random.choice(self.dim_act, [self.nenvs, self.n_agents])
            else:
                random_actions = CategoricalDistribution(tf.convert_to_tensor(avail_actions)).stochastic_sample().numpy()
            if np.random.rand() < self.egreedy:
                return hidden_state, random_actions
            else:
                return hidden_state, greedy_actions

    def train(self, i_step, n_epoch=1):
        if self.egreedy >= self.end_greedy:
            self.egreedy = self.start_greedy - self.delta_egreedy * i_step
        info_train = {}
        if i_step > self.start_training:
            for i_epoch in range(n_epoch):
                sample = self.memory.sample()
                if self.use_recurrent:
                    info_train = self.learner.update_recurrent(sample)
                else:
                    info_train = self.learner.update(sample)
        info_train["epsilon-greedy"] = self.egreedy
        return info_train
