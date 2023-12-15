QTRAN_Agents
=====================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class:: 
    xuance.torch.agent.mutli_agent_rl.qtran_agents.QTRAN_Agents(config, envs, device)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function:: 
    xuance.torch.agent.mutli_agent_rl.qtran_agents.QTRAN_Agents.train(i_step)

    Train the multi-agent reinforcement learning model.

    :param i_step: The i-th step during training.
    :type i_step: int
    :return: **info_train** - the information of the training process.
    :rtype: dict

.. raw:: html

    <br><hr>

**TensorFlow:**


.. raw:: html

    <br><hr>

**MindSpore:**

.. py:class::
    xuance.mindspore.agents.mutli_agent_rl.qtran_agents.QTRAN_Agents(config, envs)

    :param config: xxxxxx.
    :type config: xxxxxx
    :param envs: xxxxxx.
    :type envs: xxxxxx

.. py:function::
     xuance.mindspore.agents.mutli_agent_rl.qtran_agents.QTRAN_Agents.act(obs_n, *rnn_hidden, avail_actions, test_mode)

    :param obs_n: The joint observations of N agents.
    :type obs_n: numpy.ndarray
    :param rnn_hidden: The hidden states of RNN.
    :type rnn_hidden: tuple(numpy.ndarray, numpy.ndarray)
    :param avail_actions: The actions mask for available actions in the environment.
    :type avail_actions: numpy.ndarray
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :return: **hidden_state**, **actions_n** - The next hidden states of RNN and the joint actions.
    :rtype: tuple(numpy.ndarray, numpy.ndarray), np.ndarray

.. py:function::
     xuance.mindspore.agents.mutli_agent_rl.qtran_agents.QTRAN_Agents.train(i_step, n_epoch)
    :param i_step: xxxxxx.
    :type i_step: xxxxxx
    :param n_epoch: xxxxxx.
    :type n_epoch: xxxxxx
    :return: xxxxxx.
    :rtype: xxxxxx

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::
  
    .. group-tab:: PyTorch
    
        .. code-block:: python

            from xuance.torch.agents import *
            from xuance.torch.agents.agents_marl import linear_decay_or_increase

            class QTRAN_Agents(MARLAgents):
                def __init__(self,
                            config: Namespace,
                            envs: DummyVecEnv_Pettingzoo,
                            device: Optional[Union[int, str, torch.device]] = None):
                    self.gamma = config.gamma

                    self.start_greedy = config.start_greedy
                    self.end_greedy = config.end_greedy
                    self.egreedy = config.start_greedy
                    self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy
                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None

                    input_representation = get_repre_in(config)
                    representation = REGISTRY_Representation[config.representation](*input_representation)
                    mixer = VDN_mixer()
                    if config.agent == "QTRAN_base":
                        qtran_net = QTRAN_base(config.dim_state[0], config.dim_act, config.qtran_net_hidden_dim,
                                            config.n_agents, config.q_hidden_size[0]).to(device)
                    elif config.agent == "QTRAN_alt":
                        qtran_net = QTRAN_alt(config.dim_state[0], config.dim_act, config.qtran_net_hidden_dim,
                                            config.n_agents, config.q_hidden_size[0]).to(device)
                    else:
                        raise ValueError("Mixer {} not recognised.".format(config.agent))

                    input_policy = get_policy_in_marl(config, representation, config.agent_keys, mixer, qtran_mixer=qtran_net)
                    policy = REGISTRY_Policy[config.policy](*input_policy)
                    optimizer = torch.optim.Adam(policy.parameters(), config.learning_rate, eps=1e-5)
                    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,
                                                                total_iters=get_total_iters(config.agent_name, config))
                    self.observation_space = envs.observation_space
                    self.action_space = envs.action_space
                    self.representation_info_shape = policy.representation.output_shapes
                    self.auxiliary_info_shape = {}

                    memory = MARL_OffPolicyBuffer(state_shape,
                                                config.obs_shape,
                                                config.act_shape,
                                                config.rew_shape,
                                                config.done_shape,
                                                envs.num_envs,
                                                config.buffer_size,
                                                config.batch_size)
                    learner = QTRAN_Learner(config, policy, optimizer, scheduler,
                                            config.device, config.model_dir, config.gamma,
                                            config.sync_frequency)

                    self.epsilon_decay = linear_decay_or_increase(config.start_greedy, config.end_greedy,
                                                                config.greedy_update_steps)
                    super(QTRAN_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                                    config.log_dir, config.model_dir)
                    self.on_policy = False

                def train(self, i_episode):
                    self.epsilon_decay.update()
                    if self.memory.can_sample(self.args.batch_size):
                        sample = self.memory.sample()
                        info_train = self.learner.update(sample)
                        return info_train
                    else:
                        return {}



    .. group-tab:: TensorFlow
    
        .. code-block:: python3



    .. group-tab:: MindSpore

        .. code-block:: python

            from xuance.mindspore.agents import *
            from xuance.mindspore.agents.agents_marl import linear_decay_or_increase


            class QTRAN_Agents(MARLAgents):
                def __init__(self,
                             config: Namespace,
                             envs: DummyVecEnv_Pettingzoo):
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
                    mixer = VDN_mixer()
                    if config.agent == "QTRAN_base":
                        qtran_net = QTRAN_base(int(config.dim_state[0]), int(config.dim_act), int(config.qtran_net_hidden_dim),
                                               config.n_agents, config.q_hidden_size[0])
                    elif config.agent == "QTRAN_alt":
                        qtran_net = QTRAN_alt(int(config.dim_state[0]), int(config.dim_act), int(config.qtran_net_hidden_dim),
                                              config.n_agents, config.q_hidden_size[0])
                    else:
                        raise ValueError("Mixer {} not recognised.".format(config.agent))
                    input_policy = get_policy_in_marl(config, representation, mixer, qtran_mixer=qtran_net)
                    policy = REGISTRY_Policy[config.policy](*input_policy,
                                                            use_recurrent=config.use_recurrent,
                                                            rnn=config.rnn)

                    scheduler = lr_decay_model(learning_rate=config.learning_rate, decay_rate=0.5,
                                               decay_steps=get_total_iters(config.agent_name, config))
                    optimizer = Adam(policy.trainable_params(), scheduler, eps=1e-5)
                    self.observation_space = envs.observation_space
                    self.action_space = envs.action_space
                    self.representation_info_shape = policy.representation.output_shapes
                    self.auxiliary_info_shape = {}

                    buffer = MARL_OffPolicyBuffer_RNN if self.use_recurrent else MARL_OffPolicyBuffer
                    input_buffer = (config.n_agents, state_shape, config.obs_shape, config.act_shape, config.rew_shape,
                                    config.done_shape, envs.num_envs, config.buffer_size, config.batch_size)
                    memory = buffer(*input_buffer, max_episode_length=envs.max_episode_length, dim_act=config.dim_act)

                    learner = QTRAN_Learner(config, policy, optimizer, scheduler,
                                            config.model_dir, config.gamma, config.sync_frequency)
                    super(QTRAN_Agents, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)
                    self.on_policy = False

                def act(self, obs_n, *rnn_hidden, avail_actions=None, test_mode=False):
                    batch_size = obs_n.shape[0]
                    agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                                 (batch_size, -1, -1))
                    obs_in = Tensor(obs_n).view(batch_size, self.n_agents, -1)
                    if self.use_recurrent:
                        batch_agents = batch_size * self.n_agents
                        hidden_state, _, greedy_actions, _ = self.policy(obs_in.view(batch_agents, 1, -1),
                                                                      agents_id.view(batch_agents, 1, -1),
                                                                      *rnn_hidden,
                                                                      avail_actions=avail_actions.reshape(batch_agents, 1, -1))
                        greedy_actions = greedy_actions.view(batch_size, self.n_agents)
                    else:
                        hidden_state, _, greedy_actions, _ = self.policy(obs_in, agents_id, avail_actions=avail_actions)
                    greedy_actions = greedy_actions.asnumpy()

                    if test_mode:
                        return hidden_state, greedy_actions
                    else:
                        if avail_actions is None:
                            random_actions = np.random.choice(self.dim_act, [self.nenvs, self.n_agents])
                        else:
                            random_actions = Categorical(torch.Tensor(avail_actions)).sample().numpy()
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
                            info_train = self.learner.update(sample)
                    info_train["epsilon-greedy"] = self.egreedy
                    return info_train
