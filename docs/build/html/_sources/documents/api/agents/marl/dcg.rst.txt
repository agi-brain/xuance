DCG_Agents
=====================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class:: 
    xuanpolicy.torch.agent.mutli_agent_rl.dcg_agents.DCG_Agents(config, envs, device)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuanpolicy.environments.vector_envs.vector_env.VecEnv
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function:: 
    xuanpolicy.torch.agent.mutli_agent_rl.dcg_agents.DCG_Agents.act(obs_n, *rnn_hidden, avail_actions=None, test_mode=False)

    Calculate joint actions for N agents according to the joint observations.

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
    xuanpolicy.torch.agent.mutli_agent_rl.dcg_agents.DCG_Agents.train(i_step)

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

            import torch.nn
            from xuanpolicy.torch.agents import *

            class DCG_Agents(MARLAgents):
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
                    repre_state_dim = representation.output_shapes['state'][0]
                    from xuanpolicy.torch.policies.coordination_graph import DCG_utility, DCG_payoff, Coordination_Graph
                    utility = DCG_utility(repre_state_dim, config.hidden_utility_dim, config.dim_act).to(device)
                    payoffs = DCG_payoff(repre_state_dim * 2, config.hidden_payoff_dim, config.dim_act, config).to(device)
                    dcgraph = Coordination_Graph(config.n_agents, config.graph_type)
                    dcgraph.set_coordination_graph(device)
                    if config.env_name == "StarCraft2":
                        action_space = config.action_space
                    else:
                        action_space = config.action_space[config.agent_keys[0]]
                    if config.agent == "DCG_S":
                        policy = REGISTRY_Policy[config.policy](action_space,
                                                                config.state_space.shape[0], representation,
                                                                utility, payoffs, dcgraph, config.hidden_bias_dim,
                                                                None, None, torch.nn.ReLU, device,
                                                                use_recurrent=config.use_recurrent,
                                                                rnn=config.rnn)
                    else:
                        policy = REGISTRY_Policy[config.policy](action_space,
                                                                config.state_space.shape[0], representation,
                                                                utility, payoffs, dcgraph, None,
                                                                None, None, torch.nn.ReLU, device,
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

                    from xuanpolicy.torch.learners.multi_agent_rl.dcg_learner import DCG_Learner
                    learner = DCG_Learner(config, policy, optimizer, scheduler,
                                        config.device, config.model_dir, config.gamma,
                                        config.sync_frequency)
                    super(DCG_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                                    config.log_dir, config.model_dir)
                    self.on_policy = False

                def act(self, obs_n, *rnn_hidden, avail_actions=None, test_mode=False):
                    batch_size = obs_n.shape[0]
                    obs_n = torch.Tensor(obs_n).to(self.device)
                    with torch.no_grad():
                        obs_in = obs_n.view(batch_size * self.n_agents, 1, -1)
                        rnn_hidden_next, hidden_states = self.learner.get_hidden_states(obs_in, *rnn_hidden)
                        greedy_actions = self.learner.act(hidden_states.view(batch_size, self.n_agents, -1),
                                                        avail_actions=avail_actions)
                    greedy_actions = greedy_actions.cpu().detach().numpy()

                    if test_mode:
                        return rnn_hidden_next, greedy_actions
                    else:
                        if avail_actions is None:
                            random_actions = np.random.choice(self.dim_act, [self.nenvs, self.n_agents])
                        else:
                            random_actions = Categorical(torch.Tensor(avail_actions)).sample().numpy()
                        if np.random.rand() < self.egreedy:
                            return rnn_hidden_next, random_actions
                        else:
                            return rnn_hidden_next, greedy_actions

                def train(self, i_step):
                    if self.egreedy >= self.end_greedy:
                        self.egreedy = self.start_greedy - self.delta_egreedy * i_step

                    if i_step > self.start_training:
                        sample = self.memory.sample()
                        if self.use_recurrent:
                            info_train = self.learner.update_recurrent(sample)
                        else:
                            info_train = self.learner.update(sample)
                        info_train["epsilon-greedy"] = self.egreedy
                        return info_train
                    else:
                        return {}



    .. group-tab:: TensorFlow
    
        .. code-block:: python3



    .. group-tab:: MindSpore

        .. code-block:: python3