IQL_Agents
=====================================

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class:: 
    xuance.torch.agent.mutli_agent_rl.iql_agents.IQL_Agents(config, envs, device)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function:: 
    xuance.torch.agent.mutli_agent_rl.iql_agents.IQL_Agents.act(obs_n, *rnn_hidden, avail_actions=None, test_mode=False)

    Calculate joint actions for N agents according to the joint observations.

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param rnn_hidden: The hidden states of RNN.
    :type rnn_hidden: tuple(np.ndarray, np.ndarray)
    :param avail_actions: The actions mask for available actions in the environment.
    :type avail_actions: np.ndarray
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :return: **hidden_state**, **actions_n** - The next hidden states of RNN and the joint actions.
    :rtype: tuple(np.ndarray, np.ndarray), np.ndarray
  
.. py:function:: 
    xuance.torch.agent.mutli_agent_rl.iql_agents.IQL_Agents.train(i_step)

    Train the multi-agent reinforcement learning model.

    :param i_step: The i-th step during training.
    :type i_step: int
    :return: **info_train** - the information of the training process.
    :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
    xuance.tensorflow.agent.mutli_agent_rl.iql_agents.IQL_Agents(config, envs, device)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function::
    xuance.tensorflow.agent.mutli_agent_rl.iql_agents.IQL_Agents.act(obs_n, *rnn_hidden, avail_actions=None, test_mode=False)

    Calculate joint actions for N agents according to the joint observations.

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param rnn_hidden: The hidden states of RNN.
    :type rnn_hidden: tuple(np.ndarray, np.ndarray)
    :param avail_actions: The actions mask for available actions in the environment.
    :type avail_actions: np.ndarray
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :return: **hidden_state**, **actions_n** - The next hidden states of RNN and the joint actions.
    :rtype: tuple(np.ndarray, np.ndarray), np.ndarray

.. py:function::
    xuance.tensorflow.agent.mutli_agent_rl.iql_agents.IQL_Agents.train(i_step, n_epoch)

    Train the multi-agent reinforcement learning model.

    :param i_step: The i-th step during training.
    :type i_step: int
    :param n_epoch: Number of training epochs.
    :type n_epoch: int
    :return: **info_train** - the information of the training process.
    :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
    xuance.mindspore.agents.mutli_agent_rl.iql_agents.IQL_Agents(config, envs)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv

.. py:function::
     xuance.mindspore.agents.mutli_agent_rl.iql_agents.IQL_Agents.act(obs_n, *rnn_hidden, avail_actions=None, test_mode=False)

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param rnn_hidden: The hidden states of RNN.
    :type rnn_hidden: tuple(np.ndarray, np.ndarray)
    :param avail_actions: The actions mask for available actions in the environment.
    :type avail_actions: np.ndarray
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :return: **hidden_state**, **actions_n** - The next hidden states of RNN and the joint actions.
    :rtype: tuple(np.ndarray, np.ndarray), np.ndarray

.. py:function::
    xuance.mindspore.agents.mutli_agent_rl.iql_agents.IQL_Agents.train(i_step, n_epoch)
    
    :param i_step: The current training step.
    :type i_step: int
    :param n_epoch: The number of training epochs.
    :type n_epoch: int
    :return: Training information.
    :rtype: dict

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::
  
    .. group-tab:: PyTorch
    
        .. code-block:: python

            from xuance.torch.agents import *


            class IQL_Agents(MARLAgents):
                """The implementation of Independent Q-Networks agents.

                Args:
                    config: the Namespace variable that provides hyper-parameters and other settings.
                    envs: the vectorized environments.
                    device: the calculating device of the model, such as CPU or GPU.
                """
                def __init__(self,
                            config: Namespace,
                            envs: DummyVecEnv_Pettingzoo,
                            device: Optional[Union[int, str, torch.device]] = None):
                    self.gamma = config.gamma
                    self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
                    self.egreedy = self.start_greedy
                    self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy

                    input_representation = get_repre_in(config)
                    self.use_recurrent = config.use_recurrent
                    if self.use_recurrent:
                        kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                                    "dropout": config.dropout,
                                    "rnn": config.rnn}
                        representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
                    else:
                        representation = REGISTRY_Representation[config.representation](*input_representation)
                    input_policy = get_policy_in_marl(config, representation)
                    policy = REGISTRY_Policy[config.policy](*input_policy,
                                                            use_recurrent=config.use_recurrent,
                                                            rnn=config.rnn)
                    optimizer = torch.optim.Adam(policy.parameters(), config.learning_rate, eps=1e-5)
                    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,
                                                                total_iters=get_total_iters(config.agent_name, config))
                    self.observation_space = envs.observation_space
                    self.action_space = envs.action_space
                    self.representation_info_shape = policy.representation.output_shapes
                    self.auxiliary_info_shape = {}

                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None

                    buffer = MARL_OffPolicyBuffer_RNN if self.use_recurrent else MARL_OffPolicyBuffer
                    input_buffer = (config.n_agents, state_shape, config.obs_shape, config.act_shape, config.rew_shape,
                                    config.done_shape, envs.num_envs, config.buffer_size, config.batch_size)
                    memory = buffer(*input_buffer, max_episode_length=envs.max_episode_length, dim_act=config.dim_act)

                    learner = IQL_Learner(config, policy, optimizer, scheduler, config.device, config.model_dir, config.gamma,
                                        config.sync_frequency)
                    super(IQL_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                                    config.log_dir, config.model_dir)
                    self.on_policy = False

                def act(self, obs_n, *rnn_hidden, avail_actions=None, test_mode=False):
                    batch_size = obs_n.shape[0]
                    agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
                    obs_in = torch.Tensor(obs_n).view([batch_size, self.n_agents, -1]).to(self.device)
                    if self.use_recurrent:
                        batch_agents = batch_size * self.n_agents
                        hidden_state, greedy_actions, _ = self.policy(obs_in.view(batch_agents, 1, -1),
                                                                    agents_id.view(batch_agents, 1, -1),
                                                                    *rnn_hidden,
                                                                    avail_actions=avail_actions.reshape(batch_agents, 1, -1))
                        greedy_actions = greedy_actions.view(batch_size, self.n_agents)
                    else:
                        hidden_state, greedy_actions, _ = self.policy(obs_in, agents_id, avail_actions=avail_actions)
                    greedy_actions = greedy_actions.cpu().detach().numpy()

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
                            if self.use_recurrent:
                                info_train = self.learner.update_recurrent(sample)
                            else:
                                info_train = self.learner.update(sample)
                    info_train["epsilon-greedy"] = self.egreedy
                    return info_train



    .. group-tab:: TensorFlow
    
        .. code-block:: python

            from xuance.tensorflow.agents import *
            from xuance.tensorflow.agents.agents_marl import linear_decay_or_increase


            class IQL_Agents(MARLAgents):
                def __init__(self,
                             config: Namespace,
                             envs: DummyVecEnv_Pettingzoo,
                             device: str = "cpu:0"):
                    self.gamma = config.gamma
                    self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
                    self.egreedy = self.start_greedy
                    self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy

                    input_representation = get_repre_in(config)
                    self.use_recurrent = config.use_recurrent
                    if self.use_recurrent:
                        kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                                      "dropout": config.dropout,
                                      "rnn": config.rnn}
                        representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
                    else:
                        representation = REGISTRY_Representation[config.representation](*input_representation)
                    input_policy = get_policy_in_marl(config, representation)
                    policy = REGISTRY_Policy[config.policy](*input_policy,
                                                            use_recurrent=config.use_recurrent,
                                                            rnn=config.rnn)
                    # lr_scheduler = tk.optimizers.schedules.ExponentialDecay(config.learning_rate, decay_steps=1000,
                    #                                                         decay_rate=0.9)
                    lr_scheduler = MyLinearLR(config.learning_rate, start_factor=1.0, end_factor=0.5,
                                              total_iters=get_total_iters(config.agent_name, config))
                    optimizer = tk.optimizers.Adam(lr_scheduler)
                    self.observation_space = envs.observation_space
                    self.action_space = envs.action_space
                    self.representation_info_shape = policy.representation.output_shapes
                    self.auxiliary_info_shape = {}

                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None

                    buffer = MARL_OffPolicyBuffer_RNN if self.use_recurrent else MARL_OffPolicyBuffer
                    input_buffer = (config.n_agents, state_shape, config.obs_shape, config.act_shape, config.rew_shape,
                                    config.done_shape, envs.num_envs, config.buffer_size, config.batch_size)
                    memory = buffer(*input_buffer, max_episode_length=envs.max_episode_length, dim_act=config.dim_act)

                    learner = IQL_Learner(config, policy, optimizer,
                                          config.device, config.model_dir, config.gamma, config.sync_frequency)
                    super(IQL_Agents, self).__init__(config, envs, policy, memory, learner, device,
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


    .. group-tab:: MindSpore

        .. code-block:: python

            from xuance.mindspore.agents import *


            class IQL_Agents(MARLAgents):
                def __init__(self,
                            config: Namespace,
                            envs: DummyVecEnv_Pettingzoo):
                    self.gamma = config.gamma
                    self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
                    self.egreedy = self.start_greedy
                    self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy

                    input_representation = get_repre_in(config)
                    self.use_recurrent = config.use_recurrent
                    if self.use_recurrent:
                        kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                                    "dropout": config.dropout,
                                    "rnn": config.rnn}
                        representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
                    else:
                        representation = REGISTRY_Representation[config.representation](*input_representation)
                    input_policy = get_policy_in_marl(config, representation)
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

                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None

                    buffer = MARL_OffPolicyBuffer_RNN if self.use_recurrent else MARL_OffPolicyBuffer
                    input_buffer = (config.n_agents, state_shape, config.obs_shape, config.act_shape, config.rew_shape,
                                    config.done_shape, envs.num_envs, config.buffer_size, config.batch_size)
                    memory = buffer(*input_buffer, max_episode_length=envs.max_episode_length, dim_act=config.dim_act)

                    learner = IQL_Learner(config, policy, optimizer, scheduler,
                                        config.model_dir, config.gamma, config.sync_frequency)
                    super(IQL_Agents, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)
                    self.on_policy = False

                def act(self, obs_n, *rnn_hidden, avail_actions=None, test_mode=False):
                    batch_size = obs_n.shape[0]
                    agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                                (batch_size, -1, -1))
                    obs_in = Tensor(obs_n).view(batch_size, self.n_agents, -1)
                    if self.use_recurrent:
                        batch_agents = batch_size * self.n_agents
                        hidden_state, greedy_actions, _ = self.policy(obs_in.view(batch_agents, 1, -1),
                                                                    agents_id.view(batch_agents, 1, -1),
                                                                    *rnn_hidden,
                                                                    avail_actions=avail_actions.reshape(batch_agents, 1, -1))
                        greedy_actions = greedy_actions.view(batch_size, self.n_agents)
                    else:
                        hidden_state, greedy_actions, _ = self.policy(obs_in, agents_id, avail_actions=avail_actions)
                    greedy_actions = greedy_actions.asnumpy()

                    if test_mode:
                        return hidden_state, greedy_actions
                    else:
                        if avail_actions is None:
                            random_actions = np.random.choice(self.dim_act, [self.nenvs, self.n_agents])
                        else:
                            random_actions = Categorical(avail_actions).sample().asnumpy()
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



