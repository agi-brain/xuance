MFAC
=====================================

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
    xuance.torch.agent.mutli_agent_rl.mfac_agents.MFAC_Agents(config, envs, device)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function::
    xuance.torch.agent.mutli_agent_rl.mfac_agents.MFAC_Agents.act(obs_n, test_mode, act_mean=None, agent_mask=None)

    Calculate joint actions for N agents according to the joint observations.

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :param act_mean: The current mean actions.
    :type act_mean: np.ndarray
    :param agent_mask: The agent mask variables of the environments.
    :type agent_mask: np.ndarray
    :return: **hidden_state**, **actions_n**, **act_mean_current** - The next hidden states of RNN, the joint actions, and the current mean actions.
    :rtype: tuple(np.ndarray, np.ndarray), np.ndarray, np.ndarray

.. py:function::
    xuance.torch.agent.mutli_agent_rl.mfac_agents.MFAC_Agents.values(obs, actions_mean)

    Train the multi-agent reinforcement learning model.

    :param obs: The observation variables.
    :type obs: np.ndarray
    :param actions_mean: The mean values of actions.
    :type actions_mean: Tensor
    :return: hidden states and critic values.
    :rtype: tuple

.. py:function::
    xuance.torch.agent.mutli_agent_rl.mfac_agents.MFAC_Agents.train(i_step, kwargs)

    Train the multi-agent reinforcement learning model.

    :param i_step: The i-th step during training.
    :type i_step: int
    :param kwargs: The other arguments.
    :type kwargs: dict
    :return: **info_train** - the information of the training process.
    :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
    xuance.tensorflow.agent.mutli_agent_rl.mfac_agents.MFAC_Agents(config, envs, device)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function::
    xuance.tensorflow.agent.mutli_agent_rl.mfac_agents.MFAC_Agents.act(obs_n, test_mode, act_mean=None, agent_mask=None)

    Calculate joint actions for N agents according to the joint observations.

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :param act_mean: The current mean actions.
    :type act_mean: np.ndarray
    :param agent_mask: The agent mask variables of the environments.
    :type agent_mask: np.ndarray
    :return: **hidden_state**, **actions_n**, **act_mean_current** - The next hidden states of RNN, the joint actions, and the current mean actions.
    :rtype: tuple(np.ndarray, np.ndarray), np.ndarray, np.ndarray

.. py:function::
    xuance.tensorflow.agent.mutli_agent_rl.mfac_agents.MFAC_Agents.values(obs, actions_mean)

    Train the multi-agent reinforcement learning model.

    :param obs: The observation variables.
    :type obs: np.ndarray
    :param actions_mean: The mean values of actions.
    :type actions_mean: Tensor
    :return: hidden states and critic values.
    :rtype: tuple

.. py:function::
    xuance.tensorflow.agent.mutli_agent_rl.mfac_agents.MFAC_Agents.train(i_step, kwargs)

    Train the multi-agent reinforcement learning model.

    :param i_step: The i-th step during training.
    :type i_step: int
    :param kwargs: The other arguments.
    :type kwargs: dict
    :return: **info_train** - the information of the training process.
    :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
    xuance.mindspore.agent.mutli_agent_rl.mfac_agents.MFAC_Agents(config, envs)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv

.. py:function::
    xuance.mindspore.agent.mutli_agent_rl.mfac_agents.MFAC_Agents.act(obs_n, test_mode, act_mean=None, agent_mask=None)

    Calculate joint actions for N agents according to the joint observations.

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :param act_mean: The current mean actions.
    :type act_mean: np.ndarray
    :param agent_mask: The agent mask variables of the environments.
    :type agent_mask: np.ndarray
    :return: **hidden_state**, **actions_n**, **act_mean_current** - The next hidden states of RNN, the joint actions, and the current mean actions.
    :rtype: tuple(np.ndarray, np.ndarray), np.ndarray, np.ndarray

.. py:function::
    xuance.mindspore.agent.mutli_agent_rl.mfac_agents.MFAC_Agents.values(obs, actions_mean)

    Train the multi-agent reinforcement learning model.

    :param obs: The observation variables.
    :type obs: np.ndarray
    :param actions_mean: The mean values of actions.
    :type actions_mean: Tensor
    :return: hidden states and critic values.
    :rtype: tuple

.. py:function::
    xuance.mindspore.agent.mutli_agent_rl.mfac_agents.MFAC_Agents.train(i_step, kwargs)

    Train the multi-agent reinforcement learning model.

    :param i_step: The i-th step during training.
    :type i_step: int
    :param kwargs: The other arguments.
    :type kwargs: dict
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


            class MFAC_Agents(MARLAgents):
                """The implementation of Mean-Field AC agents.

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
                    self.n_envs = envs.num_envs
                    self.n_size = config.buffer_size
                    self.n_epoch = config.n_epoch
                    self.n_minibatch = config.n_minibatch
                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None

                    input_representation = get_repre_in(config)
                    representation = REGISTRY_Representation[config.representation](*input_representation)
                    input_policy = get_policy_in_marl(config, representation, config.agent_keys)
                    policy = REGISTRY_Policy[config.policy](*input_policy, gain=config.gain)
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
                    memory = MeanField_OnPolicyBuffer(config.n_agents,
                                                    state_shape,
                                                    config.obs_shape,
                                                    config.act_shape,
                                                    config.rew_shape,
                                                    config.done_shape,
                                                    envs.num_envs,
                                                    config.buffer_size,
                                                    config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda,
                                                    prob_space=config.act_prob_shape)
                    self.buffer_size = memory.buffer_size
                    self.batch_size = self.buffer_size // self.n_minibatch
                    learner = MFAC_Learner(config, policy, optimizer, scheduler,
                                        config.device, config.model_dir, config.gamma)
                    super(MFAC_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                                    config.log_dir, config.model_dir)
                    self.on_policy = True

                def act(self, obs_n, test_mode, act_mean=None, agent_mask=None):
                    batch_size = len(obs_n)
                    agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
                    obs_n = torch.Tensor(obs_n).to(self.device)

                    _, dists = self.policy(obs_n, agents_id)
                    acts = dists.stochastic_sample()

                    n_alive = torch.Tensor(agent_mask).sum(dim=-1).unsqueeze(-1).repeat(1, self.dim_act).to(self.device)
                    action_n_mask = torch.Tensor(agent_mask).unsqueeze(-1).repeat(1, 1, self.dim_act).to(self.device)
                    act_neighbor_onehot = self.learner.onehot_action(acts, self.dim_act) * action_n_mask
                    act_mean_current = act_neighbor_onehot.float().sum(dim=1) / n_alive
                    act_mean_current = act_mean_current.cpu().detach().numpy()

                    return acts.detach().cpu().numpy(), act_mean_current

                def values(self, obs, actions_mean):
                    batch_size = len(obs)
                    obs = torch.Tensor(obs).to(self.device)
                    actions_mean = torch.Tensor(actions_mean).to(self.device)
                    actions_mean = actions_mean.unsqueeze(1).expand(-1, self.n_agents, -1)
                    agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
                    values_n = self.policy.critic(obs, actions_mean, agents_id)
                    hidden_states = None
                    return hidden_states, values_n.detach().cpu().numpy()

                def train(self, i_step, **kwargs):
                    if self.memory.full:
                        info_train = {}
                        indexes = np.arange(self.buffer_size)
                        for _ in range(self.n_epoch):
                            np.random.shuffle(indexes)
                            for start in range(0, self.buffer_size, self.batch_size):
                                end = start + self.batch_size
                                sample_idx = indexes[start:end]
                                sample = self.memory.sample(sample_idx)
                                info_train = self.learner.update(sample)
                        self.learner.lr_decay(i_step)
                        self.memory.clear()
                        return info_train
                    else:
                        return {}


    .. group-tab:: TensorFlow

        .. code-block:: python

            from xuance.tensorflow.agents import *


            class MFAC_Agents(MARLAgents):
                def __init__(self,
                             config: Namespace,
                             envs: DummyVecEnv_Pettingzoo,
                             device: str = "cpu:0"):
                    self.gamma = config.gamma
                    self.n_envs = envs.num_envs
                    self.n_size = config.buffer_size
                    self.n_epoch = config.n_epoch
                    self.n_minibatch = config.n_minibatch
                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None

                    input_representation = get_repre_in(config)
                    representation = REGISTRY_Representation[config.representation](*input_representation)
                    input_policy = get_policy_in_marl(config, representation, config.agent_keys)
                    policy = REGISTRY_Policy[config.policy](*input_policy, gain=config.gain)
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
                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None
                    memory = MeanField_OnPolicyBuffer(config.n_agents,
                                                      state_shape,
                                                      config.obs_shape,
                                                      config.act_shape,
                                                      config.rew_shape,
                                                      config.done_shape,
                                                      envs.num_envs,
                                                      config.buffer_size,
                                                      config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda,
                                                      prob_space=config.act_prob_shape)
                    self.buffer_size = memory.buffer_size
                    self.batch_size = self.buffer_size // self.n_minibatch
                    learner = MFAC_Learner(config, policy, optimizer, config.device, config.model_dir, config.gamma)
                    super(MFAC_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                                      config.log_dir, config.model_dir)
                    self.on_policy = True

                def act(self, obs_n, test_mode, act_mean=None, agent_mask=None):
                    batch_size = len(obs_n)
                    inputs = {"obs": obs_n,
                              "ids": np.tile(np.expand_dims(np.eye(self.n_agents), 0), (batch_size, 1, 1))}
                    _, dists = self.policy(inputs)
                    acts = dists.stochastic_sample()

                    n_alive = np.expand_dims(np.sum(agent_mask, axis=-1), axis=-1).repeat(self.dim_act, axis=1)
                    action_n_mask = np.expand_dims(agent_mask, axis=-1).repeat(self.dim_act, axis=-1)
                    act_neighbor_onehot = self.learner.onehot_action(acts, self.dim_act).numpy() * action_n_mask
                    act_mean_current = np.sum(act_neighbor_onehot, axis=1) / n_alive

                    return acts.numpy(), act_mean_current

                def values(self, obs, actions_mean):
                    batch_size = len(obs)
                    agents_id = np.tile(np.expand_dims(np.eye(self.n_agents), 0), (batch_size, 1, 1))
                    agents_id = tf.convert_to_tensor(agents_id, dtype=tf.float32)
                    actions_mean = tf.repeat(tf.expand_dims(tf.convert_to_tensor(actions_mean, dtype=tf.float32), 1),
                                             repeats=self.n_agents, axis=1)
                    values_n = self.policy.critic(obs, actions_mean, agents_id)
                    hidden_states = None
                    return hidden_states, values_n.numpy()

                def train(self, i_step, **kwargs):
                    if self.memory.full:
                        info_train = {}
                        indexes = np.arange(self.buffer_size)
                        for _ in range(self.n_epoch):
                            np.random.shuffle(indexes)
                            for start in range(0, self.buffer_size, self.batch_size):
                                end = start + self.batch_size
                                sample_idx = indexes[start:end]
                                sample = self.memory.sample(sample_idx)
                                info_train = self.learner.update(sample)
                        self.learner.lr_decay(i_step)
                        self.memory.clear()
                        return info_train
                    else:
                        return {}


    .. group-tab:: MindSpore

        .. code-block:: python

            from xuance.mindspore.agents import *


            class MFAC_Agents(MARLAgents):
                def __init__(self,
                             config: Namespace,
                             envs: DummyVecEnv_Pettingzoo):
                    self.gamma = config.gamma
                    self.n_envs = envs.num_envs
                    self.n_size = config.buffer_size
                    self.n_epoch = config.n_epoch
                    self.n_minibatch = config.n_minibatch
                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None

                    input_representation = get_repre_in(config)
                    representation = REGISTRY_Representation[config.representation](*input_representation)
                    input_policy = get_policy_in_marl(config, representation, config.agent_keys)
                    policy = REGISTRY_Policy[config.policy](*input_policy, gain=config.gain)
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
                    memory = MeanField_OnPolicyBuffer(config.n_agents,
                                                      state_shape,
                                                      config.obs_shape,
                                                      config.act_shape,
                                                      config.rew_shape,
                                                      config.done_shape,
                                                      envs.num_envs,
                                                      config.buffer_size,
                                                      config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda,
                                                      prob_space=config.act_prob_shape)
                    self.buffer_size = memory.buffer_size
                    self.batch_size = self.buffer_size // self.n_minibatch
                    learner = MFAC_Learner(config, policy, optimizer, scheduler, config.model_dir, config.gamma)
                    super(MFAC_Agents, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)
                    self._concat = ops.Concat(axis=-1)
                    self.on_policy = True

                def act(self, obs_n, test_mode, act_mean=None, agent_mask=None):
                    batch_size = len(obs_n)
                    agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                                 (batch_size, -1, -1))
                    obs_n = Tensor(obs_n)

                    _, act_probs = self.policy(obs_n, agents_id)
                    acts = self.policy.actor.sample(act_probs)

                    n_alive = ops.broadcast_to(self.expand_dims(Tensor(agent_mask).sum(axis=-1), -1), (-1, int(self.dim_act)))
                    action_n_mask = ops.broadcast_to(self.expand_dims(Tensor(agent_mask), -1), (-1, -1, int(self.dim_act)))
                    act_neighbor_onehot = self.learner.onehot_action(acts, self.dim_act) * action_n_mask
                    act_mean_current = act_neighbor_onehot.sum(axis=1) / n_alive
                    act_mean_current = act_mean_current.asnumpy()

                    return acts.asnumpy(), act_mean_current

                def values(self, obs, actions_mean):
                    batch_size = len(obs)
                    actions_mean = ops.broadcast_to(Tensor(actions_mean).unsqueeze(1), (-1, self.n_agents, -1))
                    agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.int32), 0),
                                                 (batch_size, -1, -1)).astype(ms.float32)
                    values_n = self.policy.get_values(Tensor(obs), actions_mean, agents_id)
                    hidden_states = None
                    return hidden_states, values_n.asnumpy()

                def train(self, i_step, **kwargs):
                    if self.memory.full:
                        info_train = {}
                        indexes = np.arange(self.buffer_size)
                        for _ in range(self.n_epoch):
                            np.random.shuffle(indexes)
                            for start in range(0, self.buffer_size, self.batch_size):
                                end = start + self.batch_size
                                sample_idx = indexes[start:end]
                                sample = self.memory.sample(sample_idx)
                                info_train = self.learner.update(sample)
                        self.memory.clear()
                        return info_train
                    else:
                        return {}

