MFQ_Agents
=====================================

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class:: 
    xuance.torch.agent.mutli_agent_rl.mfq_agents.MFQ_Agents(config, envs, device)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function:: 
    xuance.torch.agent.mutli_agent_rl.mfq_agents.MFQ_Agents.act(obs_n, *rnn_hidden, act_mean=None, agent_mask=False, test_mode=False)

    Calculate joint actions for N agents according to the joint observations.

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param rnn_hidden: The hidden states of RNN.
    :type rnn_hidden: tuple(np.ndarray, np.ndarray)
    :param act_mean: The current mean actions.
    :type act_mean: np.ndarray
    :param agent_mask: The agent mask variables of the environments.
    :type agent_mask: np.ndarray
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :return: **hidden_state**, **actions_n**, **act_mean_current** - The next hidden states of RNN, the joint actions, and the current mean actions.
    :rtype: tuple(np.ndarray, np.ndarray), np.ndarray, np.ndarray
  
.. py:function:: 
    xuance.torch.agent.mutli_agent_rl.mfq_agents.MFQ_Agents.train(i_step)

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
    xuance.tensorflow.agent.mutli_agent_rl.mfq_agents.MFQ_Agents(config, envs, device)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function::
    xuance.tensorflow.agent.mutli_agent_rl.mfq_agents.MFQ_Agents.act(obs_n, *rnn_hidden, act_mean=None, agent_mask=False, test_mode=False)

    Calculate joint actions for N agents according to the joint observations.

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param rnn_hidden: The hidden states of RNN.
    :type rnn_hidden: tuple(np.ndarray, np.ndarray)
    :param act_mean: The current mean actions.
    :type act_mean: np.ndarray
    :param agent_mask: The agent mask variables of the environments.
    :type agent_mask: np.ndarray
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :return: **hidden_state**, **actions_n**, **act_mean_current** - The next hidden states of RNN, the joint actions, and the current mean actions.
    :rtype: tuple(np.ndarray, np.ndarray), np.ndarray, np.ndarray

.. py:function::
    xuance.tensorflow.agent.mutli_agent_rl.mfq_agents.MFQ_Agents.train(i_step, n_epoch)

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
    xuance.mindspore.agent.mutli_agent_rl.mfq_agents.MFQ_Agents(config, envs)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv

.. py:function::
    xuance.mindspore.agent.mutli_agent_rl.mfq_agents.MFQ_Agents.act(obs_n, *rnn_hidden, test_mode=False, act_mean=None, agent_mask=False)

    Calculate joint actions for N agents according to the joint observations.

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param rnn_hidden: The hidden states of RNN.
    :type rnn_hidden: tuple(np.ndarray, np.ndarray)
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :param act_mean: The current mean actions.
    :type act_mean: np.ndarray
    :param agent_mask: The agent mask variables of the environments.
    :type agent_mask: np.ndarray
    :return: **hidden_state**, **actions_n**, **act_mean_current** - The next hidden states of RNN, the joint actions, and the current mean actions.
    :rtype: tuple(np.ndarray, np.ndarray), np.ndarray, np.ndarray

.. py:function::
    xuance.mindspore.agent.mutli_agent_rl.mfq_agents.MFQ_Agents.train(i_step, n_epoch)

    Train the multi-agent reinforcement learning model.

    :param i_step: The i-th step during training.
    :type i_step: int
    :param n_epoch: Number of training epochs.
    :type n_epoch: int
    :return: **info_train** - the information of the training process.
    :rtype: dict

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::
  
    .. group-tab:: PyTorch
    
        .. code-block:: python

            from xuance.torch.agents import *


            class MFQ_Agents(MARLAgents):
                """The implementation of Mean-Field Q agents.

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
                    self.use_recurrent, self.rnn = config.use_recurrent, config.rnn
                    self.rnn_hidden = None

                    input_representation = get_repre_in(config)
                    representation = REGISTRY_Representation[config.representation](*input_representation)
                    input_policy = get_policy_in_marl(config, representation, config.agent_keys)
                    policy = REGISTRY_Policy[config.policy](*input_policy)
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
                    memory = MeanField_OffPolicyBuffer(config.n_agents,
                                                    state_shape,
                                                    config.obs_shape,
                                                    config.act_shape,
                                                    config.act_prob_shape,
                                                    config.rew_shape,
                                                    config.done_shape,
                                                    envs.num_envs,
                                                    config.buffer_size,
                                                    config.batch_size)
                    learner = MFQ_Learner(config, policy, optimizer, scheduler,
                                        config.device, config.model_dir, config.gamma,
                                        config.sync_frequency)
                    super(MFQ_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                                    config.log_dir, config.model_dir)
                    self.on_policy = False

                def act(self, obs_n, *rnn_hidden, test_mode=False, act_mean=None, agent_mask=None, avail_actions=None):
                    batch_size = obs_n.shape[0]
                    agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
                    obs_in = torch.Tensor(obs_n).to(self.device)
                    act_mean = torch.Tensor(act_mean).unsqueeze(dim=-2).repeat(1, self.n_agents, 1).to(self.device)

                    if self.use_recurrent:  # awaiting to be tested
                        batch_agents = batch_size * self.n_agents
                        hidden_state, greedy_actions, q_output = self.policy(obs_in.view(batch_agents, 1, -1),
                                                                            act_mean.view(batch_agents, 1, -1),
                                                                            agents_id.view(batch_agents, 1, -1),
                                                                            *rnn_hidden,
                                                                            avail_actions=avail_actions)
                    else:
                        hidden_state, greedy_actions, q_output = self.policy(obs_in, act_mean, agents_id)
                    n_alive = torch.Tensor(agent_mask).sum(dim=-1).unsqueeze(-1).repeat(1, self.dim_act).to(self.device)
                    action_n_mask = torch.Tensor(agent_mask).unsqueeze(-1).repeat(1, 1, self.dim_act).to(self.device)
                    act_neighbor_sample = self.policy.sample_actions(logits=q_output).to(self.device)
                    act_neighbor_onehot = self.learner.onehot_action(act_neighbor_sample, self.dim_act) * action_n_mask
                    act_mean_current = act_neighbor_onehot.float().sum(dim=1) / n_alive
                    act_mean_current = act_mean_current.cpu().detach().numpy()
                    greedy_actions = greedy_actions.cpu().detach().numpy()
                    if test_mode:
                        return hidden_state, greedy_actions, act_mean_current
                    else:
                        random_actions = np.random.choice(self.dim_act, [self.nenvs, self.n_agents])
                        if np.random.rand() < self.egreedy:
                            return hidden_state, random_actions, act_mean_current
                        else:
                            return hidden_state, greedy_actions, act_mean_current

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




    .. group-tab:: TensorFlow
    
        .. code-block:: python

            from xuance.tensorflow.agents import *
            from xuance.tensorflow.agents.agents_marl import linear_decay_or_increase


            class MFQ_Agents(MARLAgents):
                def __init__(self,
                             config: Namespace,
                             envs: DummyVecEnv_Pettingzoo,
                             device: str = "cpu:0"):
                    self.gamma = config.gamma

                    self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
                    self.egreedy = self.start_greedy
                    self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy
                    self.use_recurrent, self.rnn = config.use_recurrent, config.rnn
                    self.rnn_hidden = None

                    input_representation = get_repre_in(config)
                    representation = REGISTRY_Representation[config.representation](*input_representation)
                    input_policy = get_policy_in_marl(config, representation, config.agent_keys)
                    policy = REGISTRY_Policy[config.policy](*input_policy)
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
                    memory = MeanField_OffPolicyBuffer(config.n_agents,
                                                       state_shape,
                                                       config.obs_shape,
                                                       config.act_shape,
                                                       config.act_prob_shape,
                                                       config.rew_shape,
                                                       config.done_shape,
                                                       envs.num_envs,
                                                       config.buffer_size,
                                                       config.batch_size)
                    learner = MFQ_Learner(config, policy, optimizer,
                                          config.device, config.model_dir, config.gamma, config.sync_frequency)
                    super(MFQ_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                                     config.log_dir, config.model_dir)
                    self.on_policy = False

                def act(self, obs_n, *rnn_hidden, test_mode=False, act_mean=None, agent_mask=None, avail_actions=None):
                    batch_size = obs_n.shape[0]
                    act_mean = np.expand_dims(act_mean, axis=-2).repeat(self.n_agents, axis=1)
                    inputs = {"obs": obs_n,
                              "act_mean": act_mean,
                              "ids": np.tile(np.expand_dims(np.eye(self.n_agents), 0), (batch_size, 1, 1))}
                    _, greedy_actions, q_output = self.policy(inputs)
                    n_alive = np.expand_dims(np.sum(agent_mask, axis=-1), axis=-1).repeat(self.dim_act, axis=1)
                    action_n_mask = np.expand_dims(agent_mask, axis=-1).repeat(self.dim_act, axis=-1)
                    act_neighbor_sample = self.policy.sample_actions(logits=q_output)
                    act_neighbor_onehot = self.learner.onehot_action(act_neighbor_sample, self.dim_act).numpy() * action_n_mask
                    act_mean_current = np.sum(act_neighbor_onehot, axis=1) / n_alive
                    greedy_actions = greedy_actions.numpy()
                    if test_mode:
                        return None, greedy_actions, act_mean_current
                    else:
                        random_actions = np.random.choice(self.dim_act, [self.nenvs, self.n_agents])
                        if np.random.rand() < self.egreedy:
                            return None, random_actions, act_mean_current
                        else:
                            return None, greedy_actions, act_mean_current

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


    .. group-tab:: MindSpore

        .. code-block:: python

            from xuance.mindspore.agents import *
            from xuance.mindspore.agents.agents_marl import linear_decay_or_increase


            class MFQ_Agents(MARLAgents):
                def __init__(self,
                             config: Namespace,
                             envs: DummyVecEnv_Pettingzoo):
                    self.gamma = config.gamma

                    self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
                    self.egreedy = self.start_greedy
                    self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy
                    self.use_recurrent, self.rnn = config.use_recurrent, config.rnn
                    self.rnn_hidden = None

                    input_representation = get_repre_in(config)
                    representation = REGISTRY_Representation[config.representation](*input_representation)
                    input_policy = get_policy_in_marl(config, representation, config.agent_keys)
                    policy = REGISTRY_Policy[config.policy](*input_policy)
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
                    memory = MeanField_OffPolicyBuffer(config.n_agents,
                                                       state_shape,
                                                       config.obs_shape,
                                                       config.act_shape,
                                                       config.act_prob_shape,
                                                       config.rew_shape,
                                                       config.done_shape,
                                                       envs.num_envs,
                                                       config.buffer_size,
                                                       config.batch_size)
                    learner = MFQ_Learner(config, policy, optimizer, scheduler,
                                          config.model_dir, config.gamma, config.sync_frequency)
                    super(MFQ_Agents, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)
                    self.on_policy = False

                def act(self, obs_n, *rnn_hidden, test_mode=False, act_mean=None, agent_mask=None, avail_actions=None):
                    batch_size = obs_n.shape[0]
                    agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                                 (batch_size, -1, -1))
                    obs_in = Tensor(obs_n)
                    act_mean = ops.broadcast_to(self.expand_dims(Tensor(act_mean).astype(ms.float32), -2), (-1, self.n_agents, -1))

                    if self.use_recurrent:  # awaiting to be tested
                        batch_agents = batch_size * self.n_agents
                        hidden_state, greedy_actions, q_output = self.policy(obs_in.view(batch_agents, 1, -1),
                                                                             act_mean.view(batch_agents, 1, -1),
                                                                             agents_id.view(batch_agents, 1, -1),
                                                                             *rnn_hidden,
                                                                             avail_actions=avail_actions)
                    else:
                        hidden_state, greedy_actions, q_output = self.policy(obs_in, act_mean, agents_id)
                    n_alive = ops.broadcast_to(self.expand_dims(Tensor(agent_mask).sum(axis=-1), -1), (-1, int(self.dim_act)))
                    action_n_mask = ops.broadcast_to(self.expand_dims(Tensor(agent_mask), -1), (-1, -1, int(self.dim_act)))
                    act_neighbor_sample = self.policy.sample_actions(logits=q_output)
                    act_neighbor_onehot = self.learner.onehot_action(act_neighbor_sample, self.dim_act) * action_n_mask
                    act_mean_current = act_neighbor_onehot.sum(axis=1) / n_alive
                    act_mean_current = act_mean_current.asnumpy()
                    greedy_actions = greedy_actions.asnumpy()
                    if test_mode:
                        return hidden_state, greedy_actions, act_mean_current
                    else:
                        random_actions = np.random.choice(self.dim_act, [self.nenvs, self.n_agents])
                        if np.random.rand() < self.egreedy:
                            return hidden_state, random_actions, act_mean_current
                        else:
                            return hidden_state, greedy_actions, act_mean_current

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
