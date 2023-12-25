VDAC
=====================================

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
    xuance.torch.agent.mutli_agent_rl.vdac_agents.VDAC_Agents(config, envs)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv

.. py:function::
    xuance.torch.agent.mutli_agent_rl.vdac_agents.VDAC_Agents.act(obs_n, *rnn_hidden, avail_actions=None, state=None, test_mode=False)

    Calculate joint actions for N agents according to the joint observations.

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param rnn_hidden: The hidden states of RNN.
    :type rnn_hidden: tuple(np.ndarray, np.ndarray)
    :param avail_actions: The actions mask for available actions in the environment.
    :type avail_actions: np.ndarray
    :param state: The global state of the environments.
    :type state: np.ndarray
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :return: **hidden_state**, **actions_n**, **onehot_actions** - The next hidden states of RNN, the joint actions, and the onehot actions.
    :rtype: tuple(np.ndarray, np.ndarray), np.ndarray, np.ndarray

.. py:function::
    xuance.torch.agent.mutli_agent_rl.vdac_agents.VDAC_Agents.train(i_step, kwargs)

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
    xuance.tensorflow.agent.mutli_agent_rl.vdac_agents.VDAC_Agents(config, envs)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv

.. py:function::
    xuance.tensorflow.agent.mutli_agent_rl.vdac_agents.VDAC_Agents.act(obs_n, *rnn_hidden, avail_actions=None, state=None, test_mode=False)

    Calculate joint actions for N agents according to the joint observations.

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param rnn_hidden: The hidden states of RNN.
    :type rnn_hidden: tuple(np.ndarray, np.ndarray)
    :param avail_actions: The actions mask for available actions in the environment.
    :type avail_actions: np.ndarray
    :param state: The global state of the environments.
    :type state: np.ndarray
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :return: **hidden_state**, **actions_n**, **onehot_actions** - The next hidden states of RNN, the joint actions, and the onehot actions.
    :rtype: tuple(np.ndarray, np.ndarray), np.ndarray, np.ndarray

.. py:function::
    xuance.tensorflow.agent.mutli_agent_rl.vdac_agents.VDAC_Agents.train(i_step, kwargs)

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
    xuance.mindspore.agent.mutli_agent_rl.vdac_agents.VDAC_Agents(config, envs, device)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv
    :param device: Choose CPU or GPU to train the model.
    :type device: str

.. py:function::
    xuance.mindspore.agent.mutli_agent_rl.vdac_agents.VDAC_Agents.act(obs_n, *rnn_hidden, avail_actions=None, state=None, test_mode=False)

    Calculate joint actions for N agents according to the joint observations.

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param rnn_hidden: The hidden states of RNN.
    :type rnn_hidden: tuple(np.ndarray, np.ndarray)
    :param avail_actions: The actions mask for available actions in the environment.
    :type avail_actions: np.ndarray
    :param state: The global state of the environments.
    :type state: np.ndarray
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :return: A tuple that includes the next hidden states of RNN, the joint actions, and the onehot actions.
    :rtype: tuple

.. py:function::
    xuance.mindspore.agent.mutli_agent_rl.vdac_agents.VDAC_Agents.train(i_step, kwargs)

    Train the multi-agent reinforcement learning model.

    :param i_step: The i-th step during training.
    :type i_step: int
    :param kwargs: The other arguments.
    :type kwargs: dict
    :return: The information of the training process.
    :rtype: dict

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

    .. group-tab:: PyTorch

        .. code-block:: python

            from xuance.torch.agents import *


            class VDAC_Agents(MARLAgents):
                """The implementation of VDAC agents.

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
                    self.n_size = config.n_size
                    self.n_epoch = config.n_epoch
                    self.n_minibatch = config.n_minibatch
                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None

                    input_representation = get_repre_in(config)
                    self.use_recurrent = config.use_recurrent
                    # create representation for actor
                    kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                                "dropout": config.dropout,
                                "rnn": config.rnn} if self.use_recurrent else {}
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
                                                            use_recurrent=config.use_recurrent,
                                                            rnn=config.rnn,
                                                            gain=config.gain)
                    optimizer = torch.optim.Adam(policy.parameters(),
                                                lr=config.learning_rate, eps=1e-5,
                                                weight_decay=config.weight_decay)
                    self.observation_space = envs.observation_space
                    self.action_space = envs.action_space
                    self.auxiliary_info_shape = {}

                    buffer = MARL_OnPolicyBuffer_RNN if self.use_recurrent else MARL_OnPolicyBuffer
                    input_buffer = (config.n_agents, config.state_space.shape, config.obs_shape, config.act_shape, config.rew_shape,
                                    config.done_shape, envs.num_envs, config.n_size,
                                    config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda)
                    memory = buffer(*input_buffer, max_episode_length=envs.max_episode_length, dim_act=config.dim_act)
                    self.buffer_size = memory.buffer_size
                    self.batch_size = self.buffer_size // self.n_minibatch

                    learner = VDAC_Learner(config, policy, optimizer, None, config.device, config.model_dir, config.gamma)
                    super(VDAC_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                                    config.log_dir, config.model_dir)
                    self.share_values = True if config.rew_shape[0] == 1 else False
                    self.on_policy = True

                def act(self, obs_n, *rnn_hidden, avail_actions=None, state=None, test_mode=False):
                    batch_size = len(obs_n)
                    agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
                    obs_in = torch.Tensor(obs_n).view([batch_size, self.n_agents, -1]).to(self.device)
                    if state is not None:
                        state = torch.Tensor(state).to(self.device)
                    if self.use_recurrent:
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
                        hidden_state, dists, values_tot = self.policy(obs_in, agents_id,
                                                                    avail_actions=avail_actions,
                                                                    state=state)
                        actions = dists.stochastic_sample()
                        values_tot = values_tot.reshape([batch_size, self.n_agents, 1])
                    return hidden_state, actions.detach().cpu().numpy(), values_tot.detach().cpu().numpy()

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
                                if self.use_recurrent:
                                    info_train = self.learner.update_recurrent(sample)
                                else:
                                    info_train = self.learner.update(sample)
                        self.learner.lr_decay(i_step)
                        self.memory.clear()
                        return info_train
                    else:
                        return {}


    .. group-tab:: TensorFlow

        .. code-block:: python

            from xuance.tensorflow.agents import *


            class VDAC_Agents(MARLAgents):
                def __init__(self,
                             config: Namespace,
                             envs: DummyVecEnv_Pettingzoo,
                             device: str = "cpu:0"):
                    self.gamma = config.gamma
                    self.n_envs = envs.num_envs
                    self.n_size = config.n_size
                    self.n_epoch = config.n_epoch
                    self.n_minibatch = config.n_minibatch
                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None

                    input_representation = get_repre_in(config)
                    self.use_recurrent = config.use_recurrent
                    # create representation for actor
                    kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                                  "dropout": config.dropout,
                                  "rnn": config.rnn} if self.use_recurrent else {}
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
                                                            use_recurrent=config.use_recurrent,
                                                            rnn=config.rnn,
                                                            gain=config.gain)
                    lr_scheduler = MyLinearLR(config.learning_rate, start_factor=1.0, end_factor=0.5,
                                              total_iters=get_total_iters(config.agent_name, config))
                    optimizer = tk.optimizers.Adam(lr_scheduler)
                    self.observation_space = envs.observation_space
                    self.action_space = envs.action_space
                    self.auxiliary_info_shape = {}

                    buffer = MARL_OnPolicyBuffer_RNN if self.use_recurrent else MARL_OnPolicyBuffer
                    input_buffer = (config.n_agents, config.state_space.shape, config.obs_shape, config.act_shape, config.rew_shape,
                                    config.done_shape, envs.num_envs, config.n_size,
                                    config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda)
                    memory = buffer(*input_buffer, max_episode_length=envs.max_episode_length, dim_act=config.dim_act)
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
                    if self.use_recurrent:
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
                        for _ in range(self.n_epoch):
                            np.random.shuffle(indexes)
                            for start in range(0, self.buffer_size, self.batch_size):
                                end = start + self.batch_size
                                sample_idx = indexes[start:end]
                                sample = self.memory.sample(sample_idx)
                                if self.use_recurrent:
                                    info_train = self.learner.update_recurrent(sample)
                                else:
                                    info_train = self.learner.update(sample)
                        self.learner.lr_decay(i_step)
                        self.memory.clear()
                        return info_train
                    else:
                        return {}


    .. group-tab:: MindSpore

        .. code-block:: python

            from xuance.mindspore.agents import *


            class VDAC_Agents(MARLAgents):
                def __init__(self,
                             config: Namespace,
                             envs: DummyVecEnv_Pettingzoo):
                    self.gamma = config.gamma
                    self.n_envs = envs.num_envs
                    self.n_size = config.n_size
                    self.n_epoch = config.n_epoch
                    self.n_minibatch = config.n_minibatch
                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None

                    input_representation = get_repre_in(config)
                    self.use_recurrent = config.use_recurrent
                    # create representation for actor
                    kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                                  "dropout": config.dropout,
                                  "rnn": config.rnn} if self.use_recurrent else {}
                    representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
                    # create policy
                    if config.mixer == "VDN":
                        mixer = VDN_mixer()
                    elif config.mixer == "QMIX":
                        mixer = QMIX_mixer(config.dim_state[0], config.hidden_dim_mixing_net, config.hidden_dim_hyper_net,
                                           config.n_agents)
                    elif config.mixer == "Independent":
                        mixer = None
                    else:
                        raise f"Mixer named {config.mixer} is not defined!"
                    input_policy = get_policy_in_marl(config, representation, mixer=mixer)
                    policy = REGISTRY_Policy[config.policy](*input_policy,
                                                            use_recurrent=config.use_recurrent,
                                                            rnn=config.rnn,
                                                            gain=config.gain)
                    scheduler = lr_decay_model(learning_rate=config.learning_rate, decay_rate=0.5,
                                               decay_steps=get_total_iters(config.agent_name, config))
                    optimizer = nn.Adam(policy.trainable_params(), scheduler, eps=1e-5)
                    self.observation_space = envs.observation_space
                    self.action_space = envs.action_space
                    self.auxiliary_info_shape = {}

                    buffer = MARL_OnPolicyBuffer_RNN if self.use_recurrent else MARL_OnPolicyBuffer
                    input_buffer = (config.n_agents, config.state_space.shape, config.obs_shape, config.act_shape, config.rew_shape,
                                    config.done_shape, envs.num_envs, config.n_size,
                                    config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda)
                    memory = buffer(*input_buffer, max_episode_length=envs.max_episode_length, dim_act=config.dim_act)
                    self.buffer_size = memory.buffer_size
                    self.batch_size = self.buffer_size // self.n_minibatch

                    learner = VDAC_Learner(config, policy, optimizer, scheduler, config.model_dir, config.gamma)
                    super(VDAC_Agents, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)
                    self.share_values = True if config.rew_shape[0] == 1 else False
                    self.on_policy = True

                def act(self, obs_n, *rnn_hidden, avail_actions=None, state=None, test_mode=False):
                    batch_size = len(obs_n)
                    agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                                 (batch_size, -1, -1))
                    obs_in = Tensor(obs_n).view(batch_size, self.n_agents, -1)
                    if state is not None:
                        state = Tensor(state)
                    if self.use_recurrent:
                        batch_agents = batch_size * self.n_agents
                        hidden_state, act_probs, values_tot = self.policy(obs_in.view(batch_agents, 1, -1),
                                                                          agents_id.unsqueeze(2),
                                                                          *rnn_hidden,
                                                                          avail_actions=avail_actions[:, :, np.newaxis],
                                                                          state=state.unsqueeze(2))
                        actions = self.policy.actor.sample(act_probs)
                        actions = actions.reshape(batch_size, self.n_agents)
                        values_tot = values_tot.reshape([batch_size, self.n_agents, 1])
                    else:
                        hidden_state, act_probs, values_tot = self.policy(obs_in, agents_id,
                                                                          avail_actions=avail_actions,
                                                                          state=state)
                        actions = self.policy.actor.sample(act_probs)
                    return hidden_state, actions.asnumpy(), values_tot.asnumpy()

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
                                if self.use_recurrent:
                                    info_train = self.learner.update_recurrent(sample)
                                else:
                                    info_train = self.learner.update(sample)
                        self.learner.lr_decay(i_step)
                        self.memory.clear()
                        return info_train
                    else:
                        return {}


