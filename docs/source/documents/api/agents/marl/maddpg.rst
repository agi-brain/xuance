MADDPG_Agents
=====================================

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class:: 
    xuance.torch.agent.mutli_agent_rl.maddpg_agents.MADDPG_Agents(config, envs, device)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function:: 
    xuance.torch.agent.mutli_agent_rl.maddpg_agents.MADDPG_Agents.act(obs_n, test_mode=False)

    Calculate joint actions for N agents according to the joint observations.

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param test_mode: Choose if add noises on the output actions. If True, output actions directly, else output actions with noises.
    :type test_mode: bool
    :return: **actions** - The joint actions of N agents.
    :rtype: np.ndarray
  
.. py:function:: 
    xuance.torch.agent.mutli_agent_rl.maddpg_agents.MADDPG_Agents.train(i_episode)

    Train the multi-agent reinforcement learning model.

    :param i_episode: The i-th episode during training.
    :type i_episode: int
    :return: **info_train** - the information of the training process.
    :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
    xuance.tensorflow.agent.mutli_agent_rl.maddpg_agents.MADDPG_Agents(config, envs, device)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function::
    xuance.tensorflow.agent.mutli_agent_rl.maddpg_agents.MADDPG_Agents.act(obs_n, test_mode=False)

    Calculate joint actions for N agents according to the joint observations.

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param test_mode: Choose if add noises on the output actions. If True, output actions directly, else output actions with noises.
    :type test_mode: bool
    :return: **actions** - The joint actions of N agents.
    :rtype: np.ndarray

.. py:function::
    xuance.tensorflow.agent.mutli_agent_rl.maddpg_agents.MADDPG_Agents.train(i_episode)

    Train the multi-agent reinforcement learning model.

    :param i_episode: The i-th episode during training.
    :type i_episode: int
    :return: **info_train** - the information of the training process.
    :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
    xuance.mindspore.agents.mutli_agent_rl.maddpg_agents.MADDPG_Agents(config, envs)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv

.. py:function::
     xuance.mindspore.agents.mutli_agent_rl.maddpg_agents.MADDPG_Agents.act(obs_n, test_mode)

    :param obs_n: The joint observations of N agents.
    :type obs_n: np.ndarray
    :param test_mode: is True for selecting greedy actions, is False for selecting epsilon-greedy actions.
    :type test_mode: bool
    :return: Hidden state and selected actions.
    :rtype: tuple

.. py:function::
     xuance.mindspore.agents.mutli_agent_rl.maddpg_agents.MADDPG_Agents.train(i_episode)

    :param i_episode: Current episode index.
    :type i_episode: int
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


            class MADDPG_Agents(MARLAgents):
                """The implementation of MADDPG agents.

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

                    input_representation = get_repre_in(config)
                    representation = REGISTRY_Representation[config.representation](*input_representation)
                    input_policy = get_policy_in_marl(config, representation, config.agent_keys)
                    policy = REGISTRY_Policy[config.policy](*input_policy)
                    optimizer = [torch.optim.Adam(policy.parameters_actor, config.lr_a, eps=1e-5),
                                torch.optim.Adam(policy.parameters_critic, config.lr_c, eps=1e-5)]
                    scheduler = [torch.optim.lr_scheduler.LinearLR(optimizer[0], start_factor=1.0, end_factor=0.5,
                                                                total_iters=config.running_steps),
                                torch.optim.lr_scheduler.LinearLR(optimizer[1], start_factor=1.0, end_factor=0.5,
                                                                total_iters=config.running_steps)]
                    self.observation_space = envs.observation_space
                    self.action_space = envs.action_space
                    self.representation_info_shape = policy.representation.output_shapes
                    self.auxiliary_info_shape = {}

                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None
                    memory = MARL_OffPolicyBuffer(config.n_agents,
                                                state_shape,
                                                config.obs_shape,
                                                config.act_shape,
                                                config.rew_shape,
                                                config.done_shape,
                                                envs.num_envs,
                                                config.buffer_size,
                                                config.batch_size)
                    learner = MADDPG_Learner(config, policy, optimizer, scheduler,
                                            config.device, config.model_dir, config.gamma)
                    super(MADDPG_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                                        config.log_dir, config.model_dir)
                    self.on_policy = False

                def act(self, obs_n, test_mode):
                    batch_size = len(obs_n)
                    agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
                    _, actions = self.policy(torch.Tensor(obs_n), agents_id)
                    actions = actions.cpu().detach().numpy()
                    if test_mode:
                        return None, actions
                    else:
                        actions += np.random.normal(0, self.args.sigma, size=actions.shape)
                        return None, actions

                def train(self, i_episode):
                    sample = self.memory.sample()
                    info_train = self.learner.update(sample)
                    return info_train




    .. group-tab:: TensorFlow
    
        .. code-block:: python

            from xuance.tensorflow.agents import *


            class MADDPG_Agents(MARLAgents):
                def __init__(self,
                             config: Namespace,
                             envs: DummyVecEnv_Pettingzoo,
                             device: str = "cpu:0"):
                    self.gamma = config.gamma

                    input_representation = get_repre_in(config)
                    representation = REGISTRY_Representation[config.representation](*input_representation)
                    input_policy = get_policy_in_marl(config, representation, config.agent_keys)
                    policy = REGISTRY_Policy[config.policy](*input_policy)
                    lr_scheduler = [MyLinearLR(config.lr_a, start_factor=1.0, end_factor=0.5,
                                               total_iters=get_total_iters(config.agent_name, config)),
                                    MyLinearLR(config.lr_c, start_factor=1.0, end_factor=0.5,
                                               total_iters=get_total_iters(config.agent_name, config))]
                    optimizer = [tk.optimizers.Adam(lr_scheduler[0]),
                                 tk.optimizers.Adam(lr_scheduler[1])]
                    self.observation_space = envs.observation_space
                    self.action_space = envs.action_space
                    self.representation_info_shape = policy.representation.output_shapes
                    self.auxiliary_info_shape = {}

                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None
                    memory = MARL_OffPolicyBuffer(config.n_agents,
                                                  state_shape,
                                                  config.obs_shape,
                                                  config.act_shape,
                                                  config.rew_shape,
                                                  config.done_shape,
                                                  envs.num_envs,
                                                  config.buffer_size,
                                                  config.batch_size)
                    learner = MADDPG_Learner(config, policy, optimizer,
                                             config.device, config.model_dir, config.gamma)
                    super(MADDPG_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                                        config.log_dir, config.model_dir)
                    self.on_policy = False

                def act(self, obs_n, test_mode):
                    batch_size = len(obs_n)
                    with tf.device(self.device):
                        agents_id = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))
                        inputs_policy = {"obs": tf.convert_to_tensor(obs_n), "ids": agents_id}
                        _, actions = self.policy(inputs_policy)
                    actions = actions.numpy()
                    if test_mode:
                        return None, actions
                    else:
                        actions += np.random.normal(0, self.args.sigma, size=actions.shape)
                        return None, actions

                def train(self, i_episode):
                    sample = self.memory.sample()
                    info_train = self.learner.update(sample)
                    return info_train


    .. group-tab:: MindSpore

        .. code-block:: python

            from xuance.mindspore.agents import *


            class MADDPG_Agents(MARLAgents):
                def __init__(self,
                             config: Namespace,
                             envs: DummyVecEnv_Pettingzoo):
                    self.gamma = config.gamma

                    input_representation = get_repre_in(config)
                    representation = REGISTRY_Representation[config.representation](*input_representation)
                    input_policy = get_policy_in_marl(config, representation, config.agent_keys)
                    policy = REGISTRY_Policy[config.policy](*input_policy)
                    scheduler = [lr_decay_model(learning_rate=config.lr_a, decay_rate=0.5,
                                                decay_steps=get_total_iters(config.agent_name, config)),
                                 lr_decay_model(learning_rate=config.lr_c, decay_rate=0.5,
                                                decay_steps=get_total_iters(config.agent_name, config))]
                    optimizer = [Adam(policy.parameters_actor, scheduler[0], eps=1e-5),
                                 Adam(policy.parameters_critic, scheduler[1], eps=1e-5)]
                    self.observation_space = envs.observation_space
                    self.action_space = envs.action_space
                    self.representation_info_shape = policy.representation.output_shapes
                    self.auxiliary_info_shape = {}

                    if config.state_space is not None:
                        config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
                    else:
                        config.dim_state, state_shape = None, None
                    memory = MARL_OffPolicyBuffer(config.n_agents,
                                                  state_shape,
                                                  config.obs_shape,
                                                  config.act_shape,
                                                  config.rew_shape,
                                                  config.done_shape,
                                                  envs.num_envs,
                                                  config.buffer_size,
                                                  config.batch_size)
                    learner = MADDPG_Learner(config, policy, optimizer, scheduler, config.model_dir, config.gamma)
                    super(MADDPG_Agents, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)
                    self.on_policy = False

                def act(self, obs_n, test_mode):
                    batch_size = len(obs_n)
                    agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                                 (batch_size, -1, -1))
                    _, actions = self.policy(Tensor(obs_n), agents_id)
                    actions = actions.asnumpy()
                    if not test_mode:
                        actions += np.random.normal(0, self.args.sigma, size=actions.shape)
                    return None, actions

                def train(self, i_episode):
                    sample = self.memory.sample()
                    info_train = self.learner.update(sample)
                    return info_train
