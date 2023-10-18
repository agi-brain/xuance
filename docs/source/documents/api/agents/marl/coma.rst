COMA_Agents
=====================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class:: 
    xuanpolicy.torch.agent.mutli_agent_rl.coma_agents.COMA_Agents(config, envs, device)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuanpolicy.environments.vector_envs.vector_env.VecEnv
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function:: 
    xuanpolicy.torch.agent.mutli_agent_rl.coma_agents.COMA_Agents.act(obs_n, *rnn_hidden, avail_actions=None, state=None, test_mode=False)

    Calculate joint actions for N agents according to the joint observations.

    :param obs_n: The joint observations of N agents.
    :type obs_n: numpy.ndarray
    :param rnn_hidden: The hidden states of RNN.
    :type rnn_hidden: tuple(numpy.ndarray, numpy.ndarray)
    :param avail_actions: The actions mask for available actions in the environment.
    :type avail_actions: numpy.ndarray
    :param state: The global state of the environments.
    :type state: numpy.ndarray
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :return: **hidden_state**, **actions_n**, **onehot_actions** - The next hidden states of RNN, the joint actions, and the onehot actions.
    :rtype: tuple(numpy.ndarray, numpy.ndarray), np.ndarray, np.ndarray
  
.. py:function:: 
    xuanpolicy.torch.agent.mutli_agent_rl.coma_agents.COMA_Agents.train(i_step)

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

.. raw:: html

    <br><hr>

源码
-----------------

.. tabs::
  
    .. group-tab:: PyTorch
    
        .. code-block:: python3

            import torch
            from xuanpolicy.torch.agents import *
            from xuanpolicy.torch.agents.agents_marl import linear_decay_or_increase

            class COMA_Agents(MARLAgents):
                def __init__(self,
                            config: Namespace,
                            envs: DummyVecEnv_Pettingzoo,
                            device: Optional[Union[int, str, torch.device]] = None):
                    self.gamma = config.gamma
                    self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
                    self.egreedy = self.start_greedy
                    self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy

                    self.n_envs = envs.num_envs
                    self.n_size = config.n_size
                    self.n_epoch = config.n_epoch
                    self.n_minibatch = config.n_minibatch
                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape[0], config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None

                    # create representation for COMA actor
                    input_representation = get_repre_in(config)
                    self.use_recurrent = config.use_recurrent
                    self.use_global_state = config.use_global_state
                    kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                                "dropout": config.dropout,
                                "rnn": config.rnn} if self.use_recurrent else {}
                    representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
                    # create policy
                    input_policy = get_policy_in_marl(config, representation)
                    policy = REGISTRY_Policy[config.policy](*input_policy,
                                                            use_recurrent=config.use_recurrent,
                                                            rnn=config.rnn,
                                                            gain=config.gain,
                                                            use_global_state=self.use_global_state,
                                                            dim_state=config.dim_state)
                    optimizer = [torch.optim.Adam(policy.parameters_actor, config.learning_rate_actor, eps=1e-5),
                                torch.optim.Adam(policy.parameters_critic, config.learning_rate_critic, eps=1e-5)]
                    scheduler = [torch.optim.lr_scheduler.LinearLR(optimizer[0], start_factor=1.0, end_factor=0.5,
                                                                total_iters=get_total_iters(config.agent_name, config)),
                                torch.optim.lr_scheduler.LinearLR(optimizer[1], start_factor=1.0, end_factor=0.5,
                                                                total_iters=get_total_iters(config.agent_name, config))]
                    self.observation_space = envs.observation_space
                    self.action_space = envs.action_space
                    self.representation_info_shape = policy.representation.output_shapes
                    self.auxiliary_info_shape = {}

                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None
                    config.act_onehot_shape = config.act_shape + tuple([config.dim_act])

                    buffer = COMA_Buffer_RNN if self.use_recurrent else COMA_Buffer
                    input_buffer = (config.n_agents, config.state_space.shape, config.obs_shape, config.act_shape, config.rew_shape,
                                    config.done_shape, envs.num_envs, config.n_size,
                                    config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda)
                    memory = buffer(*input_buffer, max_episode_length=envs.max_episode_length,
                                    dim_act=config.dim_act, td_lambda=config.td_lambda)
                    self.buffer_size = memory.buffer_size
                    self.batch_size = self.buffer_size // self.n_minibatch

                    learner = COMA_Learner(config, policy, optimizer, scheduler,
                                        config.device, config.model_dir, config.gamma, config.sync_frequency)

                    super(COMA_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                                    config.log_dir, config.model_dir)
                    self.on_policy = True

                def act(self, obs_n, *rnn_hidden, avail_actions=None, test_mode=False):
                    batch_size = len(obs_n)
                    agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
                    obs_in = torch.Tensor(obs_n).view([batch_size, self.n_agents, -1]).to(self.device)
                    epsilon = 0.0 if test_mode else self.egreedy
                    if self.use_recurrent:
                        batch_agents = batch_size * self.n_agents
                        hidden_state, action_probs = self.policy(obs_in.view(batch_agents, 1, -1),
                                                                agents_id.view(batch_agents, 1, -1),
                                                                *rnn_hidden,
                                                                avail_actions=avail_actions.reshape(batch_agents, 1, -1),
                                                                epsilon=epsilon)
                        action_probs = action_probs.view(batch_size, self.n_agents, self.dim_act)
                    else:
                        hidden_state, action_probs = self.policy(obs_in, agents_id,
                                                                avail_actions=avail_actions,
                                                                epsilon=epsilon)
                    picked_actions = Categorical(action_probs).sample()
                    onehot_actions = self.learner.onehot_action(picked_actions, self.dim_act)
                    return hidden_state, picked_actions.detach().cpu().numpy(), onehot_actions.detach().cpu().numpy()

                def values(self, obs_n, *rnn_hidden, state=None, actions_n=None, actions_onehot=None):
                    batch_size = len(obs_n)
                    # build critic input
                    obs_n = torch.Tensor(obs_n).to(self.device)
                    actions_n = torch.Tensor(actions_n).unsqueeze(-1).to(self.device)
                    actions_in = torch.Tensor(actions_onehot).unsqueeze(1).to(self.device)
                    actions_in = actions_in.view(batch_size, 1, -1).repeat(1, self.n_agents, 1)
                    agent_mask = 1 - torch.eye(self.n_agents, device=self.device)
                    agent_mask = agent_mask.view(-1, 1).repeat(1, self.dim_act).view(self.n_agents, -1)
                    actions_in = actions_in * agent_mask.unsqueeze(0)
                    if self.use_global_state:
                        state = torch.Tensor(state).unsqueeze(1).to(self.device).repeat(1, self.n_agents, 1)
                        critic_in = torch.concat([state, obs_n, actions_in], dim=-1)
                    else:
                        critic_in = torch.concat([obs_n, actions_in])
                    # get critic values
                    hidden_state, values_n = self.policy.get_values(critic_in, target=True)

                    target_values = values_n.gather(-1, actions_n.long())
                    return hidden_state, target_values.detach().cpu().numpy()

                def train(self, i_step):
                    if self.egreedy >= self.end_greedy:
                        self.egreedy = self.start_greedy - self.delta_egreedy * i_step
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
                                    info_train = self.learner.update_recurrent(sample, self.egreedy)
                                else:
                                    info_train = self.learner.update(sample, self.egreedy)
                        self.memory.clear()
                        info_train["epsilon-greedy"] = self.egreedy
                        return info_train
                    else:
                        return {"epsilon-greedy": self.egreedy}




    .. group-tab:: TensorFlow
    
        .. code-block:: python3


    .. group-tab:: MindSpore

        .. code-block:: python3