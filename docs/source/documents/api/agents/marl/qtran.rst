QTRAN_Agents
=====================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class:: 
    xuanpolicy.torch.agent.mutli_agent_rl.qtran_agents.QTRAN_Agents(config, envs, device)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuanpolicy.environments.vector_envs.vector_env.VecEnv
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function:: 
    xuanpolicy.torch.agent.mutli_agent_rl.qtran_agents.QTRAN_Agents.train(i_step)

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

            from xuanpolicy.torch.agents import *
            from xuanpolicy.torch.agents.agents_marl import linear_decay_or_increase

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

        .. code-block:: python3