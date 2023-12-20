Deterministic-MARL
===================================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.policies.deterministic_marl.BasicQhead(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.BasicQhead.forward(x)

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx


.. py:class::
  xuance.torch.policies.deterministic_marl.BasicQnetwork(action_space, n_agents, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.BasicQnetwork.forward(observation, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.BasicQnetwork.target_Q(observation, agent_ids, *rnn_hidden)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :type *rnn_hidden: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.BasicQnetwork.copy_target()

  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic_marl.MFQnetwork(action_space, n_agents, representation, hidden_sizes, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.MFQnetwork.forward(observation, actions_mean, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions_mean: The mean values of actions.
  :type actions_mean: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MFQnetwork.sample_actions(logits)

  :param logits: The logits for categorical distributions.
  :type logits: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MFQnetwork.target_Q(observation, actions_mean, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions_mean: The mean values of actions.
  :type actions_mean: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MFQnetwork.copy_target()

  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic_marl.MixingQnetwork(action_space, n_agents, representation, mixer, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.forward(observation, agent_ids, *rnn_hidden, avail_actions)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :type *rnn_hidden: Tensor
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.target_Q(observation, agent_ids, *rnn_hidden)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :type *rnn_hidden: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.Q_tot(q, states)

  :param q: xxxxxx.
  :type q: xxxxxx
  :param states: xxxxxx.
  :type gstates: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.target_Q_tot(q, states)

  :param q: xxxxxx.
  :type q: xxxxxx
  :param states: xxxxxx.
  :type gstates: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.copy_target()

  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic_marl.Weighted_MixingQnetwork(action_space, n_agents, representation, mixer, ff_mixer, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
  :param ff_mixer: xxxxxx.
  :type ff_mixer: xxxxxx
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.Weighted_MixingQnetwork.q_centralized(observation, agent_ids, *rnn_hidden)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :type *rnn_hidden: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Weighted_MixingQnetwork.target_q_centralized(observation, agent_ids, *rnn_hidden)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :type *rnn_hidden: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Weighted_MixingQnetwork.copy_target()

  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic_marl.Qtran_MixingQnetwork(action_space, n_agents, representation, mixer, qtran_mixer, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
  :param qtran_mixer: xxxxxx.
  :type qtran_mixer: xxxxxx
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.Qtran_MixingQnetwork.forward(observation, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Qtran_MixingQnetwork.target_Q(observation, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Qtran_MixingQnetwork.copy_target()

  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic_marl.DCG_policy(action_space, global_state_dim, representation, utility, payoffs, dcgraph, hidden_size_bias, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param global_state_dim: xxxxxx.
  :type global_state_dim: xxxxxx
  :param representation: The representation module.
  :type representation: nn.Module
  :param utility: xxxxxx.
  :type utility: xxxxxx
  :param payoffs: xxxxxx.
  :type payoffs: xxxxxx
  :param hidden_size_bias: xxxxxx.
  :type hidden_size_bias: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.DCG_policy.forward(observation, agent_ids, *rnn_hidden, avail_actions)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :type *rnn_hidden: Tensor
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.DCG_policy.copy_target()

  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic_marl.ActorNet(state_dim, n_agents, action_space, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.ActorNet.forward()

  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic_marl.CriticNet(independent, state_dim, n_agents, action_dim, hidden_sizes, normalize, initialize, activation, device)

  :param independent: xxxxxx.
  :type independent: xxxxxx
  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.ACriticNet.forward()

  :return: None.
  :rtype: xxxxxx


.. py:class::
  xuance.torch.policies.deterministic_marl.Basic_DDPG_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.Basic_DDPG_policy.forward(observation, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: None.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Basic_DDPG_policy.critic(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Basic_DDPG_policy.target_critic(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Basic_DDPG_policy.soft_update(tau)

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic_marl.MADDPG_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.MADDPG_policy.critic(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MADDPG_policy.target_critic(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic_marl.MATD3_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.MATD3_policy.Qpolicy(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MATD3_policy.Qtarget(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MATD3_policy.Qaction(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MATD3_policy.soft_update()

  :return: None.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.BasicQhead(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.BasicQhead.call(x)

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx


.. py:class::
  xuance.tensorflow.policies.deterministic_marl.BasicQnetwork(action_space, n_agents, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.BasicQnetwork.call(inputs, rnn_hidden)

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.BasicQnetwork.target_Q(inputs)

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.BasicQnetwork.trainable_param()

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.BasicQnetwork.copy_target()

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.MFQnetwork(action_space, n_agents, representation, hidden_sizes, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MFQnetwork.call(inputs)

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MFQnetwork.sample_actions(logits)

  :param logits: The logits for categorical distributions.
  :type logits: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MFQnetwork.target_Q(observation, actions_mean, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions_mean: The mean values of actions.
  :type actions_mean: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MFQnetwork.copy_target()

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.MixingQnetwork(action_space, n_agents, representation, mixer, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MixingQnetwork.call(inputs, *rnn_hidden)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :type *rnn_hidden: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MixingQnetwork.target_Q(inputs)

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MixingQnetwork.Q_tot(q, states)

  :param q: xxxxxx.
  :type q: xxxxxx
  :param states: xxxxxx.
  :type gstates: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MixingQnetwork.target_Q_tot(q, states)

  :param q: xxxxxx.
  :type q: xxxxxx
  :param states: xxxxxx.
  :type gstates: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MixingQnetwork.copy_target()

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.Weighted_MixingQnetwork(action_space, n_agents, representation, mixer, ff_mixer, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
  :param ff_mixer: xxxxxx.
  :type ff_mixer: xxxxxx
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Weighted_MixingQnetwork.q_centralized(inputs)

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Weighted_MixingQnetwork.target_q_centralized(inputs)

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Weighted_MixingQnetwork.copy_target()

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.Qtran_MixingQnetwork(action_space, n_agents, representation, mixer, qtran_mixer, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
  :param qtran_mixer: xxxxxx.
  :type qtran_mixer: xxxxxx
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Qtran_MixingQnetwork.call(inputs)

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Qtran_MixingQnetwork.target_Q(inputs)

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Qtran_MixingQnetwork.copy_target()

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.DCG_policy(action_space, global_state_dim, representation, utility, payoffs, dcgraph, hidden_size_bias, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param global_state_dim: xxxxxx.
  :type global_state_dim: xxxxxx
  :param representation: The representation module.
  :type representation: nn.Module
  :param utility: xxxxxx.
  :type utility: xxxxxx
  :param payoffs: xxxxxx.
  :type payoffs: xxxxxx
  :param hidden_size_bias: xxxxxx.
  :type hidden_size_bias: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.DCG_policy.call(inputs)

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.DCG_policy.copy_target()

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.ActorNet(state_dim, n_agents, action_space, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.ActorNet.call(x)

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.CriticNet(independent, state_dim, n_agents, action_dim, hidden_sizes, normalize, initialize, activation, device)

  :param independent: xxxxxx.
  :type independent: xxxxxx
  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.CriticNet.call(x)

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: None.
  :rtype: xxxxxx


.. py:class::
  xuance.tensorflow.policies.deterministic_marl.Basic_DDPG_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Basic_DDPG_policy.call(inputs)

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: None.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Basic_DDPG_policy.critic(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Basic_DDPG_policy.target_critic(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Basic_DDPG_policy.target_actor(inputs)

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: None.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Basic_DDPG_policy.soft_update(tau)

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.MADDPG_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MADDPG_policy.critic(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MADDPG_policy.target_critic(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.Attention_CriticNet(independent, state_dim, n_agents, action_dim, hidden_sizes, normalize, initialize, activation, device)

  :param independent: xxxxxx.
  :type independent: xxxxxx
  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Attention_CriticNet.call(x)

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.AttentionCritic(independent, state_dim, n_agents, action_dim, hidden_sizes, norm_in, attend_heads)

  :param independent: xxxxxx.
  :type independent: xxxxxx
  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param norm_in: xxxxxx.
  :type norm_in: xxxxxx
  :param attend_heads: xxxxxx.
  :type attend_heads: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.AttentionCritic.shared_parameters()

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.AttentionCritic.scale_shared_grads()

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.AttentionCritic.call(inps, agents=None, return_q=True, return_all_q=False,
             regularize=False, return_attend=False, logger=None, niter=0)

  :param inps: xxxxxx.
  :type inps: xxxxxx
  :param agents: xxxxxx.
  :type agents: xxxxxx
  :param return_q: xxxxxx.
  :type return_q: xxxxxx
  :param return_all_q: xxxxxx.
  :type return_all_q: xxxxxx
  :param regularize: xxxxxx.
  :type regularize: xxxxxx
  :param return_attend: xxxxxx.
  :type return_attend: xxxxxx
  :param logger: xxxxxx.
  :type logger: xxxxxx
  :param niter: xxxxxx.
  :type niter: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.MAAC_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MAAC_policy.critic(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MAAC_policy.target_critic(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.MATD3_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MATD3_policy.call(inputs)

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MATD3_policy.critic(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MATD3_policy.target_critic(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MATD3_policy.Qaction(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MATD3_policy.target_actor(inputs)

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MATD3_policy.soft_update(tau)

  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. raw:: html

    <br><hr>

**MindSpore:**

.. py:class::
  xuance.mindspore.policies.deterministic_marl.BasicQhead(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic_marl.BasicQhead.construct(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.deterministic_marl.BasicQnetwork(action_space, n_agents, representation, hidden_size, normalize, initialize, activation, kwargs)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param kwargs: xxxxxx.
  :type kwargs: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.BasicQnetwork.construct(observation, agent_ids, rnn_hidden, avail_actions)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.BasicQnetwork.target_Q(observation, agent_ids, rnn_hidden)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.BasicQnetwork.trainable_params(recurse)

  xxxxxx.

  :param recurse: xxxxxx.
  :type recurse: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.BasicQnetwork.copy_target()

  xxxxxx.

.. py:class::
  xuance.mindspore.policies.deterministic_marl.MFQnetwork(action_space, n_agents, representation, hidden_size, normalize, initialize, activation)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param representation: The representation module.
  :type representation: nn.Module
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MFQnetwork.construct(observation, actions_mean, agent_ids)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions_mean: The mean values of actions.
  :type actions_mean: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MFQnetwork.sample_actions(logits)

  xxxxxx.

  :param logits: The logits for categorical distributions.
  :type logits: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MFQnetwork.target_Q(observation, actions_mean, agent_ids)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param actions_mean: The mean values of actions.
  :type actions_mean: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MFQnetwork.copy_target()

  xxxxxx.

.. py:class::
  xuance.mindspore.policies.deterministic_marl.MixingQnetwork(action_space, n_agents, representation, mixer, hidden_size, normalize, initialize, activation, kwargs)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param kwargs: xxxxxx.
  :type kwargs: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MixingQnetwork.construct(observation, agent_ids, rnn_hidden, avail_actions)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MixingQnetwork.target_Q(observation, agent_ids, rnn_hidden, avail_actions)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MixingQnetwork.Q_tot(q, state)

  xxxxxx.

  :param q: xxxxxx.
  :type q: xxxxxx
  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MixingQnetwork.target_Q_tot(q, state)

  xxxxxx.

  :param q: xxxxxx.
  :type q: xxxxxx
  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MixingQnetwork.trainable_params(recurse)

  xxxxxx.

  :param recurse: xxxxxx.
  :type recurse: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MixingQnetwork.copy_target()

  xxxxxx.

.. py:class::
  xuance.mindspore.policies.deterministic_marl.Weighted_MixingQnetwork(action_space, n_agents, representation, mixer, ff_mixer, hidden_size, normalize, initialize, activation, kwargs)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
  :param ff_mixer: xxxxxx.
  :type ff_mixer: xxxxxx
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param kwargs: xxxxxx.
  :type kwargs: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Weighted_MixingQnetwork.q_centralized(observation, agent_ids, rnn_hidden)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Weighted_MixingQnetwork.target_q_centralized(observation, agent_ids, rnn_hidden)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Weighted_MixingQnetwork.copy_target()

  xxxxxx.

.. py:class::
  xuance.mindspore.policies.deterministic_marl.Qtran_MixingQnetwork(action_space, n_agents, representation, mixer, qtran_mixer, hidden_size, normalize, initialize, activation, kwargs)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
  :param qtran_mixer: xxxxxx.
  :type qtran_mixer: xxxxxx
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param kwargs: xxxxxx.
  :type kwargs: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Qtran_MixingQnetwork.construct(observation, agent_ids, rnn_hidden, avail_actions)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Qtran_MixingQnetwork.target_Q(observation, agent_ids, rnn_hidden)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Weighted_MixingQnetwork.copy_target()

  xxxxxx.
  

.. py:class::
  xuance.mindspore.policies.deterministic_marl.DCG_policy(action_space, global_state_dim, representation, utility, payoffs, dcgraph, hidden_size_bias, normalize, initialize, activation, kwargs)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param global_state_dim: xxxxxx.
  :type global_state_dim: xxxxxx
  :param representation: The representation module.
  :type representation: nn.Module
  :param utility: xxxxxx.
  :type utility: xxxxxx
  :param payoffs: xxxxxx.
  :type payoffs: xxxxxx
  :param dcgraph: xxxxxx.
  :type dcgraph: xxxxxx
  :param hidden_size_bias: xxxxxx.
  :type hidden_size_bias: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param kwargs: xxxxxx.
  :type kwargs: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.DCG_policy.construct(observation, agent_ids, rnn_hidden, avail_actions)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.DCG_policy.copy_target()

  xxxxxx.

.. py:class::
  xuance.mindspore.policies.deterministic_marl.ActorNet(state_dim, n_agents, action_dim, hidden_sizes, normalize, initialize, activation)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic_marl.ActorNet.construct(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.deterministic_marl.CriticNet(independent, state_dim, n_agents, action_dim, hidden_sizes, normalize, initialize, activation)

  :param independent: xxxxxx.
  :type independent: xxxxxx
  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic_marl.CriticNet.construct(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.deterministic_marl.Basic_DDPG_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Basic_DDPG_policy.construct(observation, agent_ids)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Basic_DDPG_policy.critic(observation, action, agent_ids)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: xxxxxx.
  :type action: xxxxxx
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Basic_DDPG_policy.target_critic(observation, action, agent_ids)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: xxxxxx.
  :type action: xxxxxx
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Basic_DDPG_policy.target_actor(observation, agent_ids)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Basic_DDPG_policy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. py:class::
  xuance.mindspore.policies.deterministic_marl.MADDPG_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MADDPG_policy.construct(observation, agent_ids)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MADDPG_policy.critic(observation, action, agent_ids)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: xxxxxx.
  :type action: xxxxxx
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MADDPG_policy.target_critic(observation, action, agent_ids)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: xxxxxx.
  :type action: xxxxxx
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MADDPG_policy.target_actor(observation, agent_ids)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MADDPG_policy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. py:class::
  xuance.mindspore.policies.deterministic_marl.MATD3_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  :param action_space: The action space of the environment.
  :type action_space: Box, Discrete, etc
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MATD3_policy.Qpolicy(observation, action, agent_ids)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: xxxxxx.
  :type action: xxxxxx
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MATD3_policy.Qtarget(observation, action, agent_ids)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: xxxxxx.
  :type action: xxxxxx
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MATD3_policy.Qaction_A(observation, action, agent_ids)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: xxxxxx.
  :type action: xxxxxx
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MATD3_policy.Qaction_B(observation, action, agent_ids)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: xxxxxx.
  :type action: xxxxxx
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MATD3_policy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        import copy

        import numpy as np
        import torch

        from xuance.torch.policies import *
        from xuance.torch.utils import *
        from xuance.torch.representations import Basic_Identical
        from gymnasium.spaces.box import Box as Box_pettingzoo
        from gymnasium import spaces as spaces_pettingzoo


        class BasicQhead(nn.Module):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         n_agents: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(BasicQhead, self).__init__()
                layers_ = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
                    layers_.extend(mlp)
                layers_.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
                self.model = nn.Sequential(*layers_)

            def forward(self, x: torch.Tensor):
                return self.model(x)


        class BasicQnetwork(nn.Module):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: nn.Module,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None,
                         **kwargs):
                super(BasicQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(self.representation)
                self.representation_info_shape = self.representation.output_shapes
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                             hidden_size, normalize, initialize, activation, device)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)

            def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                        *rnn_hidden: torch.Tensor, avail_actions=None):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
                evalQ = self.eval_Qhead(q_inputs)
                if avail_actions is not None:
                    avail_actions = torch.Tensor(avail_actions)
                    evalQ_detach = evalQ.clone().detach()
                    evalQ_detach[avail_actions == 0] = -9999999
                    argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
                else:
                    argmax_action = evalQ.argmax(dim=-1, keepdim=False)
                return rnn_hidden, argmax_action, evalQ

            def target_Q(self, observation: torch.Tensor, agent_ids: torch.Tensor, *rnn_hidden: torch.Tensor):
                if self.use_rnn:
                    outputs = self.target_representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.target_representation(observation)
                    rnn_hidden = None
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
                return rnn_hidden, self.target_Qhead(q_inputs)

            def copy_target(self):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
                    tp.data.copy_(ep)


        class MFQnetwork(nn.Module):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: nn.Module,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(MFQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(self.representation)
                self.representation_info_shape = self.representation.output_shapes

                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                             n_agents, hidden_size, normalize, initialize, activation, device)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)

            def forward(self, observation: torch.Tensor, actions_mean: torch.Tensor, agent_ids: torch.Tensor):
                outputs = self.representation(observation)
                q_inputs = torch.concat([outputs['state'], actions_mean, agent_ids], dim=-1)
                evalQ = self.eval_Qhead(q_inputs)
                argmax_action = evalQ.argmax(dim=-1, keepdim=False)
                return outputs, argmax_action, evalQ

            def sample_actions(self, logits: torch.Tensor):
                dist = Categorical(logits=logits)
                return dist.sample()

            def target_Q(self, observation: torch.Tensor, actions_mean: torch.Tensor, agent_ids: torch.Tensor):
                outputs = self.target_representation(observation)
                q_inputs = torch.concat([outputs['state'], actions_mean, agent_ids], dim=-1)
                return self.target_Qhead(q_inputs)

            def copy_target(self):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
                    tp.data.copy_(ep)


        class MixingQnetwork(nn.Module):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: nn.Module,
                         mixer: Optional[VDN_mixer] = None,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None,
                         **kwargs):
                super(MixingQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(self.representation)
                self.representation_info_shape = self.representation.output_shapes
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                             hidden_size, normalize, initialize, activation, device)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)
                self.eval_Qtot = mixer
                self.target_Qtot = copy.deepcopy(self.eval_Qtot)

            def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                        *rnn_hidden: torch.Tensor, avail_actions=None):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
                evalQ = self.eval_Qhead(q_inputs)
                if avail_actions is not None:
                    avail_actions = torch.Tensor(avail_actions)
                    evalQ_detach = evalQ.clone().detach()
                    evalQ_detach[avail_actions == 0] = -9999999
                    argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
                else:
                    argmax_action = evalQ.argmax(dim=-1, keepdim=False)

                return rnn_hidden, argmax_action, evalQ

            def target_Q(self, observation: torch.Tensor, agent_ids: torch.Tensor, *rnn_hidden: torch.Tensor):
                if self.use_rnn:
                    outputs = self.target_representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.target_representation(observation)
                    rnn_hidden = None
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
                return rnn_hidden, self.target_Qhead(q_inputs)

            def Q_tot(self, q, states=None):
                return self.eval_Qtot(q, states)

            def target_Q_tot(self, q, states=None):
                return self.target_Qtot(q, states)

            def copy_target(self):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.eval_Qtot.parameters(), self.target_Qtot.parameters()):
                    tp.data.copy_(ep)


        class Weighted_MixingQnetwork(MixingQnetwork):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: nn.Module,
                         mixer: Optional[VDN_mixer] = None,
                         ff_mixer: Optional[QMIX_FF_mixer] = None,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None,
                         **kwargs):
                super(Weighted_MixingQnetwork, self).__init__(action_space, n_agents, representation, mixer, hidden_size,
                                                              normalize, initialize, activation, device, **kwargs)
                self.eval_Qhead_centralized = copy.deepcopy(self.eval_Qhead)
                self.target_Qhead_centralized = copy.deepcopy(self.eval_Qhead_centralized)
                self.q_feedforward = ff_mixer
                self.target_q_feedforward = copy.deepcopy(self.q_feedforward)

            def q_centralized(self, observation: torch.Tensor, agent_ids: torch.Tensor, *rnn_hidden: torch.Tensor):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                else:
                    outputs = self.representation(observation)
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
                return self.eval_Qhead_centralized(q_inputs)

            def target_q_centralized(self, observation: torch.Tensor, agent_ids: torch.Tensor, *rnn_hidden: torch.Tensor):
                if self.use_rnn:
                    outputs = self.target_representation(observation, *rnn_hidden)
                else:
                    outputs = self.target_representation(observation)
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
                return self.target_Qhead_centralized(q_inputs)

            def copy_target(self):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.eval_Qtot.parameters(), self.target_Qtot.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.eval_Qhead_centralized.parameters(), self.target_Qhead_centralized.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.q_feedforward.parameters(), self.target_q_feedforward.parameters()):
                    tp.data.copy_(ep)


        class Qtran_MixingQnetwork(nn.Module):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: nn.Module,
                         mixer: Optional[VDN_mixer] = None,
                         qtran_mixer: Optional[QTRAN_base] = None,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(Qtran_MixingQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(self.representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                             hidden_size, normalize, initialize, activation, device)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)
                self.qtran_net = qtran_mixer
                self.target_qtran_net = copy.deepcopy(qtran_mixer)
                self.q_tot = mixer

            def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
                outputs = self.representation(observation)
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
                evalQ = self.eval_Qhead(q_inputs)
                argmax_action = evalQ.argmax(dim=-1, keepdim=False)
                return outputs, argmax_action, evalQ

            def target_Q(self, observation: torch.Tensor, agent_ids: torch.Tensor):
                outputs = self.target_representation(observation)
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
                return outputs, self.target_Qhead(q_inputs)

            def copy_target(self):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.qtran_net.parameters(), self.target_qtran_net.parameters()):
                    tp.data.copy_(ep)


        class DCG_policy(nn.Module):
            def __init__(self,
                         action_space: Discrete,
                         global_state_dim: int,
                         representation: nn.Module,
                         utility: Optional[nn.Module] = None,
                         payoffs: Optional[nn.Module] = None,
                         dcgraph: Optional[nn.Module] = None,
                         hidden_size_bias: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None,
                         **kwargs):
                super(DCG_policy, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(self.representation)
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.utility = utility
                self.target_utility = copy.deepcopy(self.utility)
                self.payoffs = payoffs
                self.target_payoffs = copy.deepcopy(self.payoffs)
                self.graph = dcgraph
                self.dcg_s = False
                if hidden_size_bias is not None:
                    self.dcg_s = True
                    self.bias = BasicQhead(global_state_dim, 1, 0, hidden_size_bias,
                                           normalize, initialize, activation, device)
                    self.target_bias = copy.deepcopy(self.bias)

            def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                        *rnn_hidden: torch.Tensor, avail_actions=None):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
                evalQ = self.eval_Qhead(q_inputs)
                if avail_actions is not None:
                    avail_actions = torch.Tensor(avail_actions)
                    evalQ_detach = evalQ.clone().detach()
                    evalQ_detach[avail_actions == 0] = -9999999
                    argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
                else:
                    argmax_action = evalQ.argmax(dim=-1, keepdim=False)
                return rnn_hidden, argmax_action, evalQ

            def copy_target(self):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.utility.parameters(), self.target_utility.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.payoffs.parameters(), self.target_payoffs.parameters()):
                    tp.data.copy_(ep)
                if self.dcg_s:
                    for ep, tp in zip(self.bias.parameters(), self.target_bias.parameters()):
                        tp.data.copy_(ep)


        class ActorNet(nn.Module):
            def __init__(self,
                         state_dim: int,
                         n_agents: int,
                         action_space: spaces_pettingzoo,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(ActorNet, self).__init__()
                layers = []
                input_shape = (state_dim + n_agents,)
                action_dim = action_space.shape[0]
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, nn.Sigmoid, initialize, device)[0])
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.tensor):
                return self.model(x)


        class CriticNet(nn.Module):
            def __init__(self,
                         independent: bool,
                         state_dim: int,
                         n_agents: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None
                         ):
                super(CriticNet, self).__init__()
                layers = []
                if independent:
                    input_shape = (state_dim + action_dim + n_agents,)
                else:
                    input_shape = (state_dim * n_agents + action_dim * n_agents + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.tensor):
                return self.model(x)


        class Basic_DDPG_policy(nn.Module):
            def __init__(self,
                         action_space: spaces_pettingzoo,
                         n_agents: int,
                         representation: nn.Module,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None
                         ):
                super(Basic_DDPG_policy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.n_agents = n_agents
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes

                self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, action_space,
                                          actor_hidden_size, normalize, initialize, activation, device)
                self.critic_net = CriticNet(True, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                            critic_hidden_size, normalize, initialize, activation, device)
                self.target_actor_net = copy.deepcopy(self.actor_net)
                self.target_critic_net = copy.deepcopy(self.critic_net)
                self.parameters_actor = list(self.representation.parameters()) + list(self.actor_net.parameters())
                self.parameters_critic = self.critic_net.parameters()

            def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
                outputs = self.representation(observation)
                actor_in = torch.concat([outputs['state'], agent_ids], dim=-1)
                act = self.actor_net(actor_in)
                return outputs, act

            def critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
                outputs = self.representation(observation)
                critic_in = torch.concat([outputs['state'], actions, agent_ids], dim=-1)
                return self.critic_net(critic_in)

            def target_critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
                outputs = self.representation(observation)
                critic_in = torch.concat([outputs['state'], actions, agent_ids], dim=-1)
                return self.target_critic_net(critic_in)

            def target_actor(self, observation: torch.Tensor, agent_ids: torch.Tensor):
                outputs = self.representation(observation)
                actor_in = torch.concat([outputs['state'], agent_ids], dim=-1)
                return self.target_actor_net(actor_in)

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor_net.parameters(), self.target_actor_net.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.critic_net.parameters(), self.target_critic_net.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)


        class MADDPG_policy(Basic_DDPG_policy):
            def __init__(self,
                         action_space: spaces_pettingzoo,
                         n_agents: int,
                         representation: nn.Module,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None
                         ):
                super(MADDPG_policy, self).__init__(action_space, n_agents, representation,
                                                    actor_hidden_size, critic_hidden_size,
                                                    normalize, initialize, activation, device)
                self.critic_net = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                            critic_hidden_size, normalize, initialize, activation, device)
                self.target_critic_net = copy.deepcopy(self.critic_net)
                self.parameters_critic = self.critic_net.parameters()

            def critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
                bs = observation.shape[0]
                outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
                actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
                critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
                return self.critic_net(critic_in)

            def target_critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
                bs = observation.shape[0]
                outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
                actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
                critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
                return self.target_critic_net(critic_in)


        class MATD3_policy(Basic_DDPG_policy):
            def __init__(self,
                         action_space: Space,
                         n_agents: int,
                         representation: nn.Module,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None
                         ):
                super(MATD3_policy, self).__init__(action_space, n_agents, representation,
                                                   actor_hidden_size, critic_hidden_size,
                                                   normalize, initialize, activation, device)
                self.critic_net_A = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                              critic_hidden_size, normalize, initialize, activation, device)
                self.critic_net_B = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                              critic_hidden_size, normalize, initialize, activation, device)
                self.target_critic_net_A = copy.deepcopy(self.critic_net_A)
                self.target_critic_net_B = copy.deepcopy(self.critic_net_B)
                # self.parameters_critic = self.critic_net.parameters()

            def Qpolicy(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
                bs = observation.shape[0]
                outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
                actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
                critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
                qa = self.critic_net_A(critic_in)
                qb = self.critic_net_B(critic_in)
                return outputs_n, (qa + qb) / 2.0

            def Qtarget(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
                bs = observation.shape[0]
                outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
                # noise = torch.randn_like(actions).clamp(-1, 1) * 0.1
                actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
                # noise = noise.view(bs, 1, -1).expand(-1, self.n_agents, -1)
                # actions_n = (actions_n + noise).clamp(-1, 1)
                critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
                qa = self.target_critic_net_A(critic_in)
                qb = self.target_critic_net_B(critic_in)
                min_q = torch.minimum(qa, qb)
                return outputs_n, min_q

            def Qaction(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
                bs = observation.shape[0]
                outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
                actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
                critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
                qa = self.critic_net_A(critic_in)
                qb = self.critic_net_B(critic_in)
                return outputs_n, torch.cat((qa, qb), dim=-1)

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor_net.parameters(), self.target_actor_net.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.critic_net_A.parameters(), self.target_critic_net_A.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.critic_net_B.parameters(), self.target_critic_net_B.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)



  .. group-tab:: TensorFlow

    .. code-block:: python

        from xuance.tensorflow.policies import *
        from xuance.tensorflow.utils import *
        from xuance.tensorflow.representations import Basic_Identical

        from itertools import chain


        class BasicQhead(tk.Model):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         n_agents: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(BasicQhead, self).__init__()
                layers_ = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
                    layers_.extend(mlp)
                layers_.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
                self.model = tk.Sequential(layers_)

            def call(self, x: tf.Tensor, **kwargs):
                return self.model(x)


        class BasicQnetwork(tk.Model):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0",
                         **kwargs):
                super(BasicQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(self.representation)
                self.representation_info_shape = self.representation.output_shapes
                self.obs_dim = self.representation.input_shapes[0]
                self.n_agents = n_agents
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                             hidden_size, normalize, initializer, activation, device)
                self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                               hidden_size, normalize, initializer, activation, device)
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

            def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
                observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
                IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
                if self.use_rnn:
                    outputs = self.representation(observations, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observations)
                    rnn_hidden = None
                q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
                evalQ = tf.reshape(self.eval_Qhead(q_inputs), [-1, self.n_agents, self.action_dim])
                if ('avail_actions' in kwargs.keys()) and (kwargs['avail_actions'] is not None):
                    evalQ_detach = evalQ.clone().detach()
                    avail_actions = kwargs['avail_actions']
                    evalQ_detach[avail_actions == 0] = -9999999
                    argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
                else:
                    argmax_action = tf.argmax(evalQ, axis=-1)
                return rnn_hidden, argmax_action, evalQ

            def target_Q(self, inputs: Union[np.ndarray, dict]):
                shape_obs = inputs["obs"].shape
                shape_ids = inputs["ids"].shape
                observations = tf.reshape(inputs['obs'], [-1, shape_obs[-1]])
                IDs = tf.reshape(inputs['ids'], [-1, shape_ids[-1]])
                outputs = self.target_representation(observations)
                q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
                return tf.reshape(self.target_Qhead(q_inputs), shape_obs[0:-1] + (self.action_dim,))

            def trainable_param(self):
                return self.representation.trainable_variables + self.eval_Qhead.trainable_variables

            def copy_target(self):
                self.target_representation.set_weights(self.representation.get_weights())
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


        class MFQnetwork(tk.Model):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(MFQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(self.representation)
                self.representation_info_shape = self.representation.output_shapes

                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                             n_agents, hidden_size, normalize, initializer, activation, device)
                self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                               n_agents, hidden_size, normalize, initializer, activation, device)
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

            def call(self, inputs: Union[np.ndarray, dict], **kwargs):
                observation = inputs["obs"]
                actions_mean = inputs["act_mean"]
                agent_ids = inputs["ids"]
                outputs = self.representation(observation)
                q_inputs = tf.concat([outputs['state'], actions_mean, agent_ids], axis=-1)
                evalQ = self.eval_Qhead(q_inputs)
                argmax_action = tf.argmax(evalQ, axis=-1)
                return outputs, argmax_action, evalQ

            def sample_actions(self, logits: tf.Tensor):
                dist = tfp.distributions.Categorical(logits=logits)
                return dist.sample()

            def target_Q(self, observation: tf.Tensor, actions_mean: tf.Tensor, agent_ids: tf.Tensor):
                outputs = self.target_representation(observation)
                q_inputs = tf.concat([outputs['state'], actions_mean, agent_ids], axis=-1)
                return self.target_Qhead(q_inputs)

            def copy_target(self):
                self.target_representation.set_weights(self.representation.get_weights())
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


        class MixingQnetwork(tk.Model):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         mixer: Optional[VDN_mixer] = None,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0",
                         **kwargs):
                super(MixingQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(self.representation)
                self.representation_info_shape = self.representation.output_shapes
                self.obs_dim = self.representation.input_shapes[0]
                self.n_agents = n_agents
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                             hidden_size, normalize, initializer, activation, device)
                self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                               hidden_size, normalize, initializer, activation, device)
                self.eval_Qtot = mixer
                self.target_Qtot = mixer
                # self.copy_target()
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
                self.target_Qtot.set_weights(self.eval_Qtot.get_weights())

            def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
                observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
                IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
                if self.use_rnn:
                    outputs = self.representation(observations, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observations)
                    rnn_hidden = None
                q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
                evalQ = tf.reshape(self.eval_Qhead(q_inputs), [-1, self.n_agents, self.action_dim])
                if ('avail_actions' in kwargs.keys()) and (kwargs['avail_actions'] is not None):
                    evalQ_detach = evalQ.clone().detach()
                    avail_actions = kwargs['avail_actions']
                    evalQ_detach[avail_actions == 0] = -9999999
                    argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
                else:
                    argmax_action = tf.argmax(evalQ, axis=-1)
                return rnn_hidden, argmax_action, evalQ

            def target_Q(self, inputs: Union[np.ndarray, dict]):
                shape_obs = inputs["obs"].shape
                shape_ids = inputs["ids"].shape
                observations = tf.reshape(inputs['obs'], [-1, shape_obs[-1]])
                IDs = tf.reshape(inputs['ids'], [-1, shape_ids[-1]])
                outputs = self.target_representation(observations)
                q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
                return tf.reshape(self.target_Qhead(q_inputs), shape_obs[0:-1] + (self.action_dim,))

            def Q_tot(self, q, states=None):
                return self.eval_Qtot(q, states)

            def target_Q_tot(self, q, states=None):
                return self.target_Qtot(q, states)

            def copy_target(self):
                self.target_representation.set_weights(self.representation.get_weights())
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
                self.target_Qtot.set_weights(self.eval_Qtot.get_weights())


        class Weighted_MixingQnetwork(MixingQnetwork):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         mixer: Optional[VDN_mixer] = None,
                         ff_mixer: Optional[QMIX_FF_mixer] = None,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0",
                         **kwargs):
                super(Weighted_MixingQnetwork, self).__init__(action_space, n_agents, representation, mixer, hidden_size,
                                                              normalize, initializer, activation, device, **kwargs)
                self.eval_Qhead_centralized = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                                         n_agents, hidden_size, normalize, initializer, activation, device)
                self.target_Qhead_centralized = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                                           n_agents, hidden_size, normalize, initializer, activation, device)
                self.q_feedforward = ff_mixer
                self.target_q_feedforward = ff_mixer
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
                self.target_Qtot.set_weights(self.eval_Qtot.get_weights())
                self.target_Qhead_centralized.set_weights(self.eval_Qhead_centralized.get_weights())
                self.target_q_feedforward.set_weights(self.q_feedforward.get_weights())

            def q_centralized(self, inputs: Union[np.ndarray, dict]):
                observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
                IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
                outputs = self.representation(observations)
                q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
                return tf.reshape(self.eval_Qhead_centralized(q_inputs), [-1, self.n_agents, self.action_dim])

            def target_q_centralized(self, inputs: Union[np.ndarray, dict]):
                observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
                IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
                outputs = self.target_representation(observations)
                q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
                return tf.reshape(self.target_Qhead_centralized(q_inputs), [-1, self.n_agents, self.action_dim])

            def copy_target(self):
                self.target_representation.set_weights(self.representation.get_weights())
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
                self.target_Qtot.set_weights(self.eval_Qtot.get_weights())
                self.target_Qhead_centralized.set_weights(self.eval_Qhead_centralized.get_weights())
                self.target_q_feedforward.set_weights(self.q_feedforward.get_weights())


        class Qtran_MixingQnetwork(tk.Model):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         mixer: Optional[VDN_mixer] = None,
                         qtran_mixer: Optional[QTRAN_base] = None,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0",
                         **kwargs):
                super(Qtran_MixingQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(self.representation)
                self.representation_info_shape = self.representation.output_shapes
                self.obs_dim = self.representation.input_shapes[0]
                self.hidden_state_dim = self.representation.output_shapes['state'][0]
                self.n_agents = n_agents
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                             hidden_size, normalize, initializer, activation, device)
                self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                               hidden_size, normalize, initializer, activation, device)
                self.qtran_net = qtran_mixer
                self.target_qtran_net = qtran_mixer
                self.q_tot = mixer
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
                self.target_qtran_net.set_weights(self.qtran_net.get_weights())

            def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
                observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
                IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
                outputs = self.representation(observations)
                q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
                evalQ = tf.reshape(self.eval_Qhead(q_inputs), [-1, self.n_agents, self.action_dim])
                argmax_action = tf.argmax(evalQ, axis=-1)
                return tf.reshape(outputs['state'], [-1, self.n_agents, self.hidden_state_dim]), argmax_action, evalQ

            def target_Q(self, inputs: Union[np.ndarray, dict]):
                observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
                IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
                outputs = self.target_representation(observations)
                q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
                return tf.reshape(outputs['state'], [-1, self.n_agents, self.hidden_state_dim]), self.target_Qhead(q_inputs)

            def copy_target(self):
                self.target_representation.set_weights(self.representation.get_weights())
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
                self.target_qtran_net.set_weights(self.qtran_net.get_weights())


        class DCG_policy(tk.Model):
            def __init__(self,
                         action_space: Discrete,
                         global_state_dim: int,
                         representation: Optional[Basic_Identical],
                         utility: Optional[DCG_utility] = None,
                         payoffs: Optional[DCG_payoff] = None,
                         dcgraph: Optional[Coordination_Graph] = None,
                         hidden_size_bias: Sequence[int] = None,
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0",
                         **kwargs):
                super(DCG_policy, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = representation
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.utility = utility
                self.target_utility = utility
                self.payoffs = payoffs
                self.target_payoffs = payoffs
                self.graph = dcgraph
                self.dcg_s = False
                if hidden_size_bias is not None:
                    self.dcg_s = True
                    self.bias = BasicQhead(global_state_dim, 1, 0, hidden_size_bias,
                                           normalize, initializer, activation, device)
                    self.target_bias = BasicQhead(global_state_dim, 1, 0, hidden_size_bias,
                                                  normalize, initializer, activation, device)

            def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden: torch.Tensor, **kwargs):
                observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
                IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
                outputs = self.representation(observations)
                q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
                evalQ = self.eval_Qhead(q_inputs)
                evalQ = tf.reshape(evalQ, [-1, self.n_agents, self.action_dim])
                argmax_action = tf.argmax(evalQ, axis=-1)
                return outputs, argmax_action, evalQ

            def copy_target(self):
                self.target_representation.set_weights(self.representation.get_weights())
                self.target_utility.set_weights(self.utility.get_weights())
                self.target_payoffs.set_weights(self.payoffs.get_weights())
                if self.dcg_s:
                    self.target_bias.set_weights(self.bias.get_weights())


        class ActorNet(tk.Model):
            def __init__(self,
                         state_dim: int,
                         n_agents: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(ActorNet, self).__init__()
                layers = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, tk.layers.Activation("tanh"), initializer, device)[0])
                self.model = tk.Sequential(layers)

            def call(self, x: tf.Tensor, **kwargs):
                return self.model(x)


        class CriticNet(tk.Model):
            def __init__(self,
                         independent: bool,
                         state_dim: int,
                         n_agents: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"
                         ):
                super(CriticNet, self).__init__()
                layers = []
                if independent:
                    input_shape = (state_dim + action_dim + n_agents,)
                else:
                    input_shape = (state_dim * n_agents + action_dim * n_agents + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initializer, device)[0])
                self.model = tk.Sequential(layers)

            def call(self, x: tf.Tensor, **kwargs):
                return self.model(x)


        class Basic_DDPG_policy(tk.Model):
            def __init__(self,
                         action_space: Space,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"
                         ):
                super(Basic_DDPG_policy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.n_agents = n_agents
                self.representation = representation
                self.obs_dim = self.representation.input_shapes[0]
                self.representation_info_shape = self.representation.output_shapes

                self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                          actor_hidden_size, normalize, initializer, activation, device)
                self.target_actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                                 actor_hidden_size, normalize, initializer, activation, device)
                self.critic_net = CriticNet(True, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                            critic_hidden_size, normalize, initializer, activation, device)
                self.target_critic_net = CriticNet(True, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                                   critic_hidden_size, normalize, initializer, activation, device)
                if isinstance(self.representation, Basic_Identical):
                    self.parameters_actor = self.actor_net.trainable_variables
                else:
                    self.parameters_actor = self.representation.trainable_variables + self.actor_net.trainable_variables
                self.parameters_critic = self.critic_net.trainable_variables
                self.soft_update(1.0)

            def call(self, inputs: Union[np.ndarray, dict], **kwargs):
                observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
                IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
                outputs = self.representation(observations)
                actor_in = tf.concat([outputs['state'], IDs], axis=-1)
                act = self.actor_net(actor_in)
                return outputs, tf.reshape(act, [-1, self.n_agents, self.action_dim])

            def critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
                observation = tf.reshape(observation, [-1, self.obs_dim])
                actions = tf.reshape(actions, [-1, self.action_dim])
                agent_ids = tf.reshape(agent_ids, [-1, self.n_agents])
                outputs = self.representation(observation)
                critic_in = tf.concat([outputs['state'], actions, agent_ids], axis=-1)
                return tf.reshape(self.critic_net(critic_in), [-1, self.n_agents, 1])

            def target_critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
                observation = tf.reshape(observation, [-1, self.obs_dim])
                actions = tf.reshape(actions, [-1, self.action_dim])
                agent_ids = tf.reshape(agent_ids, [-1, self.n_agents])
                outputs = self.representation(observation)
                critic_in = tf.concat([outputs['state'], actions, agent_ids], axis=-1)
                return tf.reshape(self.target_critic_net(critic_in), [-1, self.n_agents, 1])

            def target_actor(self, inputs: Union[np.ndarray, dict]):
                observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
                IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
                outputs = self.representation(observations)
                actor_in = tf.concat([outputs['state'], IDs], axis=-1)
                act = self.target_actor_net(actor_in)
                return tf.reshape(act, [-1, self.n_agents, self.action_dim])

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor_net.variables, self.target_actor_net.variables):
                    tp.assign((1 - tau) * tp + tau * ep)
                for ep, tp in zip(self.critic_net.variables, self.target_critic_net.variables):
                    tp.assign((1 - tau) * tp + tau * ep)


        class MADDPG_policy(Basic_DDPG_policy):
            def __init__(self,
                         action_space: Space,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"
                         ):
                super(MADDPG_policy, self).__init__(action_space, n_agents, representation,
                                                    actor_hidden_size, critic_hidden_size,
                                                    normalize, initializer, activation, device)
                self.critic_net = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                            critic_hidden_size, normalize, initializer, activation, device)
                self.target_critic_net = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                                   critic_hidden_size, normalize, initializer, activation, device)
                self.parameters_critic = self.critic_net.trainable_variables
                self.soft_update(1.0)

            def critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
                bs = observation.shape[0]
                outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
                outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
                actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
                critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
                return self.critic_net(critic_in)

            def target_critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
                bs = observation.shape[0]
                outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
                outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
                actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
                critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
                return self.target_critic_net(critic_in)


        class Attention_CriticNet(tk.Model):
            def __init__(self,
                         independent: bool,
                         state_dim: int,
                         n_agents: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"
                         ):
                super(Attention_CriticNet, self).__init__()
                layers = []
                if independent:
                    input_shape = (state_dim + action_dim + n_agents,)
                else:
                    input_shape = (state_dim * n_agents + action_dim * n_agents + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initializer, device)[0])
                self.model = tk.Sequential(layers)

            def call(self, x: tf.Tensor, **kwargs):
                return self.model(x)


        class AttentionCritic(tk.Model):
            def __init__(self,
                         independent: bool,
                         state_dim: int,
                         n_agent: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         norm_in=True,
                         attend_heads=1):
                super(AttentionCritic, self).__init__()
                assert (hidden_sizes[0] % attend_heads) == 0
                # if independent:
                #     input_shape = (state_dim + action_dim + n_agent, )
                # else:
                #     input_shape = (state_dim * n_agent + action_dim * n_agent + n_agent)
                self.attend_heads = attend_heads
                self.state_dim = state_dim
                self.action_dim = action_dim
                self.n_agent = n_agent
                self.critic_encoders = nn.ModuleList()
                self.critics = nn.ModuleList()
                self.state_encoders = nn.ModuleList()
                # iterate over agents
                idim = state_dim * n_agent + action_dim * n_agent + n_agent
                odim = action_dim
                encoder = tk.Sequential()
                if norm_in:
                    encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                                affine=False))
                encoder.add_module('enc_fc1', nn.Linear(idim, hidden_sizes[0]))
                encoder.add_module('enc_nl', nn.LeakyReLU())
                self.critic_encoders.append(encoder)
                critic = tk.Sequential()
                critic.add_module('critic_fc1', nn.Linear(2 * hidden_sizes[0],
                                                          hidden_sizes[0]))
                critic.add_module('critic_nl', nn.LeakyReLU())
                critic.add_module('critic_fc2', nn.Linear(hidden_sizes[0], odim))
                self.critics.append(critic)

                state_encoder = tk.Sequential()
                if norm_in:
                    state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                        state_dim * n_agent, affine=False))
                state_encoder.add_module('s_enc_fc1', nn.Linear(state_dim * n_agent,
                                                                hidden_sizes[0]))
                state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
                self.state_encoders.append(state_encoder)

                attend_dim = hidden_sizes[0] // attend_heads
                self.key_extractors = nn.ModuleList()
                self.selector_extractors = nn.ModuleList()
                self.value_extractors = nn.ModuleList()
                for i in range(attend_heads):
                    self.key_extractors.append(nn.Linear(hidden_sizes[0], attend_dim, bias=False))
                    self.selector_extractors.append(nn.Linear(hidden_sizes[0], attend_dim, bias=False))
                    self.value_extractors.append(tk.Sequential(nn.Linear(hidden_sizes[0],
                                                                         attend_dim),
                                                               nn.LeakyReLU()))

                self.shared_modules = [self.key_extractors, self.selector_extractors,
                                       self.value_extractors, self.critic_encoders]

            def shared_parameters(self):
                """
                Parameters shared across agents and reward heads
                """
                return chain(*[m.parameters() for m in self.shared_modules])

            def scale_shared_grads(self):
                """
                Scale gradients for parameters that are shared since they accumulate
                gradients from the critic loss function multiple times
                """
                for p in self.shared_parameters():
                    p.grad.data.mul_(1. / self.nagents)

            def call(self, inps, agents=None, return_q=True, return_all_q=False,
                     regularize=False, return_attend=False, logger=None, niter=0):
                if agents is None:
                    agents = range(len(self.critic_encoders))
                states = inps[:, :, :self.state_dim * self.n_agent]
                actions = inps[:, :, self.state_dim * self.n_agent: (self.state_dim + self.action_dim) * self.n_agent]
                # extract state-action encoding for each agent
                sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
                # extract state encoding for each agent that we're returning Q for
                s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
                # s_encodings = [self.state_encoders[a_i](states[:, a_i, :]) for a_i in agents]
                # extract keys for each head for each agent
                all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
                # extract sa values for each head for each agent
                all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
                # extract selectors for each head for each agent that we're returning Q for
                all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                                      for sel_ext in self.selector_extractors]

                other_all_values = [[] for _ in range(self.n_agent)]
                all_attend_logits = [[] for _ in range(self.n_agent)]
                all_attend_probs = [[] for _ in range(self.n_agent)]
                # calculate attention per head
                for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                        all_head_keys, all_head_values, all_head_selectors):
                    # iterate over agents
                    for i, a_i, selector in zip(range(self.n_agent), range(self.n_agent), curr_head_selectors[0]):
                        keys = [k for j, k in enumerate(curr_head_keys[0]) if j != a_i]
                        values = [v for j, v in enumerate(curr_head_values[0]) if j != a_i]
                        # calculate attention across agents
                        # attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),  torch.stack(keys).permute(1, 2, 0))
                        attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                                     keys[0].view(keys[0].shape[0], 1, -1))
                        # scale dot-products by size of key (from Attention is All You Need)
                        scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[0])
                        attend_weights = F.softmax(scaled_attend_logits, dim=0)
                        other_values = (values[0].view(values[0].shape[0], 1, -1) * attend_weights).sum(dim=2)
                        other_all_values[i].append(other_values)
                        all_attend_logits[i].append(attend_logits[:, :, 0])
                        all_attend_probs[i].append(attend_weights[:, :, 0])
                # calculate Q per agent
                all_rets = []
                for i, a_i in enumerate(range(1)):
                    head_entropies = [(-((probs + 1e-8).log() * probs).sum(1).mean()) for probs in all_attend_probs[i]]
                    agent_rets = []
                    critic_in = tf.concat((s_encodings[0][i], *other_all_values[i][0]), dim=0)
                    all_q = self.critics[a_i](critic_in)
                    int_acs = \
                        actions[:, a_i, a_i * self.action_dim:a_i * self.action_dim + self.action_dim].max(dim=0, keepdim=True)[
                            1]
                    q = all_q.gather(1, int_acs)
                    if return_q:
                        agent_rets.append(q)
                    if return_all_q:
                        agent_rets.append(all_q)
                    if regularize:
                        # regularize magnitude of attention logits
                        attend_mag_reg = 1e-3 * sum((logit ** 2).mean() for logit in
                                                    all_attend_logits[i])
                        regs = (attend_mag_reg,)
                        agent_rets.append(regs)
                    if return_attend:
                        agent_rets.append(np.array(all_attend_probs[i]))
                    if logger is not None:
                        logger.add_scalars('agent%i/attention' % a_i,
                                           dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                                in enumerate(head_entropies)),
                                           niter)
                    if len(agent_rets) == 1:
                        all_rets.append(agent_rets[0])
                    else:
                        all_rets.append(agent_rets)
                if len(all_rets) == 1:
                    return all_rets[0]
                else:
                    return all_rets


        class MAAC_policy(Basic_DDPG_policy):
            def __init__(self,
                         action_space: Space,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"
                         ):
                assert isinstance(action_space, Box)
                super(MAAC_policy, self).__init__(action_space, n_agents, representation,
                                                  actor_hidden_size, critic_hidden_size,
                                                  normalize, initializer, activation, device)
                self.critic_net = AttentionCritic(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                                  critic_hidden_size, norm_in=True, attend_heads=1)
                self.target_critic_net = copy.deepcopy(self.critic_net)
                self.parameters_critic = self.critic_net.parameters()

            def critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
                bs = observation.shape[0]
                outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
                actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
                critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
                return self.critic_net(critic_in)

            def target_critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
                bs = observation.shape[0]
                outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
                actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
                critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
                return self.target_critic_net(critic_in)


        class MATD3_policy(tk.Model):
            def __init__(self,
                         action_space: Space,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"
                         ):
                super(MATD3_policy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.n_agents = n_agents
                self.representation = representation
                self.obs_dim = self.representation.input_shapes[0]
                self.representation_info_shape = self.representation.output_shapes

                self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                          actor_hidden_size, normalize, initializer, activation, device)
                self.target_actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                                 actor_hidden_size, normalize, initializer, activation, device)
                self.critic_net_A = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                              critic_hidden_size, normalize, initializer, activation, device)
                self.critic_net_B = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                              critic_hidden_size, normalize, initializer, activation, device)
                self.target_critic_net_A = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                                     critic_hidden_size, normalize, initializer, activation, device)
                self.target_critic_net_B = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                                     critic_hidden_size, normalize, initializer, activation, device)
                self.soft_update(tau=1.0)
                self.critic_parameters = self.critic_net_A.trainable_variables + self.critic_net_B.trainable_variables

            def call(self, inputs: Union[np.ndarray, dict], **kwargs):
                observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
                IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
                outputs = self.representation(observations)
                actor_in = tf.concat([outputs['state'], IDs], axis=-1)
                act = self.actor_net(actor_in)
                return outputs, tf.reshape(act, [-1, self.n_agents, self.action_dim])

            def critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
                bs = observation.shape[0]
                outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
                outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
                actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
                critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
                qa = self.critic_net_A(critic_in)
                qb = self.critic_net_B(critic_in)
                return outputs_n, (qa + qb) / 2.0

            def target_critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
                bs = observation.shape[0]
                outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
                outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
                actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
                critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
                qa = self.target_critic_net_A(critic_in)
                qb = self.target_critic_net_B(critic_in)
                min_q = tf.math.minimum(qa, qb)
                return outputs_n, min_q

            def Qaction(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
                bs = observation.shape[0]
                outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
                outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
                actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
                critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
                qa = self.critic_net_A(critic_in)
                qb = self.critic_net_B(critic_in)
                return outputs_n, tf.concat((qa, qb), axis=-1)

            def target_actor(self, inputs: Union[np.ndarray, dict]):
                observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
                IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
                outputs = self.representation(observations)
                actor_in = tf.concat([outputs['state'], IDs], axis=-1)
                act = self.target_actor_net(actor_in)
                return tf.reshape(act, [-1, self.n_agents, self.action_dim])

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor_net.variables, self.target_actor_net.variables):
                    tp.assign((1 - tau) * tp + tau * ep)
                for ep, tp in zip(self.critic_net_A.variables, self.target_critic_net_A.variables):
                    tp.assign((1 - tau) * tp + tau * ep)
                for ep, tp in zip(self.critic_net_B.variables, self.target_critic_net_B.variables):
                    tp.assign((1 - tau) * tp + tau * ep)


  .. group-tab:: MindSpore

    .. code-block:: python

        import markdown.extensions.smarty

        from xuance.mindspore.policies import *
        from xuance.mindspore.utils import *
        import copy
        from xuance.mindspore.representations import Basic_Identical
        from mindspore.nn.probability.distribution import Categorical


        class BasicQhead(nn.Cell):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         n_agents: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                super(BasicQhead, self).__init__()
                layers_ = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
                    layers_.extend(mlp)
                layers_.extend(mlp_block(input_shape[0], action_dim, None, None, None)[0])
                self.model = nn.SequentialCell(*layers_)

            def construct(self, x: ms.tensor):
                return self.model(x)


        class BasicQnetwork(nn.Cell):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         **kwargs):
                super(BasicQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(self.representation)
                self.representation_info_shape = self.representation.output_shapes
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                             hidden_size, normalize, initialize, activation)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)
                self._concat = ms.ops.Concat(axis=-1)

            def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                          *rnn_hidden: torch.Tensor, avail_actions=None):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                q_inputs = self._concat([outputs['state'], agent_ids])
                evalQ = self.eval_Qhead(q_inputs)
                if avail_actions is not None:
                    evalQ_detach = copy.deepcopy(evalQ)
                    evalQ_detach[avail_actions == 0] = -9999999
                    argmax_action = evalQ_detach.argmax(axis=-1)
                else:
                    argmax_action = evalQ.argmax(axis=-1)
                return rnn_hidden, argmax_action, evalQ

            def target_Q(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: torch.Tensor):
                if self.use_rnn:
                    outputs = self.target_representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.target_representation(observation)
                    rnn_hidden = None
                q_inputs = self._concat([outputs['state'], agent_ids])
                return rnn_hidden, self.target_Qhead(q_inputs)

            def trainable_params(self, recurse=True):
                return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

            def copy_target(self):
                for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
                    tp.assign_value(ep)


        class MFQnetwork(nn.Cell):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                super(MFQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes

                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                             n_agents, hidden_size, normalize, initialize, activation)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)
                self._concat = ms.ops.Concat(axis=-1)
                self._dist = Categorical(dtype=ms.float32)

            def construct(self, observation: ms.Tensor, actions_mean: ms.Tensor, agent_ids: ms.Tensor):
                outputs = self.representation(observation)
                q_inputs = self._concat([outputs['state'], actions_mean, agent_ids])
                evalQ = self.eval_Qhead(q_inputs)
                argmax_action = evalQ.argmax(axis=-1)
                return outputs, argmax_action, evalQ

            def sample_actions(self, logits: ms.Tensor):
                return self._dist.sample(probs=logits).astype(ms.int32)

            def target_Q(self, observation: ms.Tensor, actions_mean: ms.Tensor, agent_ids: ms.Tensor):
                outputs = self.representation(observation)
                q_inputs = self._concat([outputs['state'], actions_mean, agent_ids])
                return self.target_Qhead(q_inputs)

            def copy_target(self):
                for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
                    tp.assign_value(ep)


        class MixingQnetwork(nn.Cell):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         mixer: Optional[VDN_mixer] = None,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         **kwargs):
                super(MixingQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(self.representation)
                self.representation_info_shape = self.representation.output_shapes
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                             hidden_size, normalize, initialize, activation)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)
                self.eval_Qtot = mixer
                self.target_Qtot = copy.deepcopy(self.eval_Qtot)
                self._concat = ms.ops.Concat(axis=-1)

            def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                          *rnn_hidden: torch.Tensor, avail_actions=None):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                q_inputs = self._concat([outputs['state'], agent_ids])
                evalQ = self.eval_Qhead(q_inputs)
                if avail_actions is not None:
                    evalQ_detach = copy.deepcopy(evalQ)
                    evalQ_detach[avail_actions == 0] = -9999999
                    argmax_action = evalQ_detach.argmax(axis=-1)
                else:
                    argmax_action = evalQ.argmax(axis=-1)
                return rnn_hidden, argmax_action, evalQ

            def target_Q(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: torch.Tensor):
                if self.use_rnn:
                    outputs = self.target_representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.target_representation(observation)
                    rnn_hidden = None
                q_inputs = self._concat([outputs['state'], agent_ids])
                return rnn_hidden, self.target_Qhead(q_inputs)

            def Q_tot(self, q, states=None):
                return self.eval_Qtot(q, states)

            def target_Q_tot(self, q, states=None):
                return self.target_Qtot(q, states)

            def trainable_params(self, recurse=True):
                return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

            def copy_target(self):
                for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.eval_Qtot.trainable_params(), self.target_Qtot.trainable_params()):
                    tp.assign_value(ep)


        class Weighted_MixingQnetwork(MixingQnetwork):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         mixer: Optional[VDN_mixer] = None,
                         ff_mixer: Optional[QMIX_FF_mixer] = None,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         **kwargs):
                super(Weighted_MixingQnetwork, self).__init__(action_space, n_agents, representation, mixer, hidden_size,
                                                              normalize, initialize, activation, **kwargs)
                self.eval_Qhead_centralized = copy.deepcopy(self.eval_Qhead)
                self.target_Qhead_centralized = copy.deepcopy(self.eval_Qhead_centralized)
                self.q_feedforward = ff_mixer
                self.target_q_feedforward = copy.deepcopy(self.q_feedforward)
                self._concat = ms.ops.Concat(axis=-1)

            def q_centralized(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: torch.Tensor):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                else:
                    outputs = self.representation(observation)
                q_inputs = self._concat([outputs['state'], agent_ids])
                return self.eval_Qhead_centralized(q_inputs)

            def target_q_centralized(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: torch.Tensor):
                if self.use_rnn:
                    outputs = self.target_representation(observation, *rnn_hidden)
                else:
                    outputs = self.target_representation(observation)
                q_inputs = self._concat([outputs['state'], agent_ids])
                return self.target_Qhead_centralized(q_inputs)

            def copy_target(self):
                for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.eval_Qtot.trainable_params(), self.target_Qtot.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.eval_Qhead_centralized.trainable_params(), self.target_Qhead_centralized.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.q_feedforward.trainable_params(), self.target_q_feedforward.trainable_params()):
                    tp.assign_value(ep)


        class Qtran_MixingQnetwork(nn.Cell):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         mixer: Optional[VDN_mixer] = None,
                         qtran_mixer: Optional[QTRAN_base] = None,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         **kwargs):
                super(Qtran_MixingQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(self.representation)
                self.representation_info_shape = self.representation.output_shapes
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                             hidden_size, normalize, initialize, activation)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)
                self.qtran_net = qtran_mixer
                self.target_qtran_net = copy.deepcopy(qtran_mixer)
                self.q_tot = mixer
                self._concat = ms.ops.Concat(axis=-1)

            def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                          *rnn_hidden: torch.Tensor, avail_actions=None):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                q_inputs = self._concat([outputs['state'], agent_ids])
                evalQ = self.eval_Qhead(q_inputs)
                if avail_actions is not None:
                    evalQ_detach = copy.deepcopy(evalQ)
                    evalQ_detach[avail_actions == 0] = -9999999
                    argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
                else:
                    argmax_action = evalQ.argmax(dim=-1, keepdim=False)
                return rnn_hidden, outputs['state'], argmax_action, evalQ

            def target_Q(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: torch.Tensor):
                if self.use_rnn:
                    outputs = self.target_representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.target_representation(observation)
                    rnn_hidden = None
                q_inputs = self._concat([outputs['state'], agent_ids])
                return rnn_hidden, outputs['state'], self.target_Qhead(q_inputs)

            def copy_target(self):
                for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.qtran_net.trainable_params(), self.target_qtran_net.trainable_params()):
                    tp.assign_value(ep)


        class DCG_policy(nn.Cell):
            def __init__(self,
                         action_space: Discrete,
                         global_state_dim: int,
                         representation: Optional[Basic_Identical],
                         utility: Optional[DCG_utility] = None,
                         payoffs: Optional[DCG_payoff] = None,
                         dcgraph: Optional[Coordination_Graph] = None,
                         hidden_size_bias: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         **kwargs):
                super(DCG_policy, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(self.representation)
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.utility = utility
                self.target_utility = copy.deepcopy(self.utility)
                self.payoffs = payoffs
                self.target_payoffs = copy.deepcopy(self.payoffs)
                self.graph = dcgraph
                self.dcg_s = False
                if hidden_size_bias is not None:
                    self.dcg_s = True
                    self.bias = BasicQhead(global_state_dim, 1, 0, hidden_size_bias,
                                           normalize, initialize, activation)
                    self.target_bias = copy.deepcopy(self.bias)
                self._concat = ms.ops.Concat(axis=-1)

            def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                          *rnn_hidden: torch.Tensor, avail_actions=None):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                q_inputs = self._concat([outputs['state'], agent_ids])
                evalQ = self.eval_Qhead(q_inputs)
                if avail_actions is not None:
                    evalQ_detach = copy.deepcopy(evalQ)
                    evalQ_detach[avail_actions == 0] = -9999999
                    argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
                else:
                    argmax_action = evalQ.argmax(dim=-1, keepdim=False)
                return rnn_hidden, argmax_action, evalQ

            def copy_target(self):
                for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.utility.trainable_params(), self.target_utility.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.payoffs.trainable_params(), self.target_payoffs.trainable_params()):
                    tp.assign_value(ep)
                if self.dcg_s:
                    for ep, tp in zip(self.bias.trainable_params(), self.target_bias.trainable_params()):
                        tp.assign_value(ep)


        class ActorNet(nn.Cell):
            def __init__(self,
                         state_dim: int,
                         n_agents: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                super(ActorNet, self).__init__()
                layers = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, nn.Tanh, initialize)[0])
                self.model = nn.SequentialCell(*layers)

            def construct(self, x: ms.tensor):
                return self.model(x)


        class CriticNet(nn.Cell):
            def __init__(self,
                         independent: bool,
                         state_dim: int,
                         n_agents: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                super(CriticNet, self).__init__()
                layers = []
                if independent:
                    input_shape = (state_dim + action_dim + n_agents,)
                else:
                    input_shape = (state_dim * n_agents + action_dim * n_agents + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initialize)[0])
                self.model = nn.SequentialCell(*layers)

            def construct(self, x: ms.tensor):
                return self.model(x)


        class Basic_DDPG_policy(nn.Cell):
            def __init__(self,
                         action_space: Space,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                super(Basic_DDPG_policy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.n_agents = n_agents
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes

                self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                          actor_hidden_size, normalize, initialize, activation)
                self.critic_net = CriticNet(True, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                            critic_hidden_size, normalize, initialize, activation)
                self.target_actor_net = copy.deepcopy(self.actor_net)
                self.target_critic_net = copy.deepcopy(self.critic_net)
                self.parameters_actor = self.representation.trainable_params() + self.actor_net.trainable_params()
                self.parameters_critic = self.critic_net.trainable_params()
                self._concat = ms.ops.Concat(axis=-1)

            def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor):
                outputs = self.representation(observation)
                actor_in = self._concat([outputs['state'], agent_ids])
                act = self.actor_net(actor_in)
                return outputs, act

            def critic(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
                outputs = self.representation(observation)
                critic_in = self._concat([outputs['state'], actions, agent_ids])
                return self.critic_net(critic_in)

            def target_critic(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
                outputs = self.representation(observation)
                critic_in = self._concat([outputs['state'], actions, agent_ids])
                return self.target_critic_net(critic_in)

            def target_actor(self, observation: ms.Tensor, agent_ids: ms.Tensor):
                outputs = self.representation(observation)
                actor_in = self._concat([outputs['state'], agent_ids])
                return self.target_actor_net(actor_in)

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
                    tp.assign_value((tau*ep.data+(1-tau)*tp.data))
                for ep, tp in zip(self.critic_net.trainable_params(), self.target_critic_net.trainable_params()):
                    tp.assign_value((tau*ep.data+(1-tau)*tp.data))


        class MADDPG_policy(nn.Cell):
            def __init__(self,
                         action_space: Space,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                super(MADDPG_policy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.n_agents = n_agents
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes

                self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                          actor_hidden_size, normalize, initialize, activation)
                self.critic_net = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                            critic_hidden_size, normalize, initialize, activation)
                self.target_actor_net = copy.deepcopy(self.actor_net)
                self.target_critic_net = copy.deepcopy(self.critic_net)
                self.parameters_actor = self.representation.trainable_params() + self.actor_net.trainable_params()
                self.parameters_critic = self.critic_net.trainable_params()
                self._concat = ms.ops.Concat(axis=-1)
                self._concat = ms.ops.Concat(axis=-1)
                self.broadcast_to = ms.ops.BroadcastTo((-1, self.n_agents, -1))

            def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor):
                outputs = self.representation(observation)
                actor_in = self._concat([outputs['state'], agent_ids])
                act = self.actor_net(actor_in)
                return outputs, act

            def critic(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
                bs = observation.shape[0]
                outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
                actions_n = self.broadcast_to(actions.view(bs, 1, -1))
                critic_in = self._concat([outputs_n, actions_n, agent_ids])
                return self.critic_net(critic_in)

            def target_critic(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
                bs = observation.shape[0]
                outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
                actions_n = self.broadcast_to(actions.view(bs, 1, -1))
                critic_in = self._concat([outputs_n, actions_n, agent_ids])
                return self.target_critic_net(critic_in)

            def target_actor(self, observation: ms.Tensor, agent_ids: ms.Tensor):
                outputs = self.representation(observation)
                actor_in = self._concat([outputs['state'], agent_ids])
                return self.target_actor_net(actor_in)

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
                    tp.assign_value((tau*ep.data+(1-tau)*tp.data))
                for ep, tp in zip(self.critic_net.trainable_params(), self.target_critic_net.trainable_params()):
                    tp.assign_value((tau*ep.data+(1-tau)*tp.data))


        class MATD3_policy(Basic_DDPG_policy):
            def __init__(self,
                         action_space: Space,
                         n_agents: int,
                         representation: Optional[Basic_Identical],
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(MATD3_policy, self).__init__(action_space, n_agents, representation,
                                                   actor_hidden_size, critic_hidden_size,
                                                   normalize, initialize, activation)
                self.critic_net_A = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                            critic_hidden_size, normalize, initialize, activation)
                self.critic_net_B = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                              critic_hidden_size, normalize, initialize, activation)
                self.parameters_actor = self.representation.trainable_params() + self.actor_net.trainable_params()
                self.parameters_critic_A = self.critic_net_A.trainable_params()
                self.parameters_critic_B = self.critic_net_B.trainable_params()

                self.target_critic_net_A = copy.deepcopy(self.critic_net_A)
                self.target_critic_net_B = copy.deepcopy(self.critic_net_B)
                self.broadcast_to = ms.ops.BroadcastTo((-1, self.n_agents, -1))

            def Qpolicy(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
                bs = observation.shape[0]
                outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
                critic_in = self._concat([outputs_n, actions, agent_ids])
                qa = self.critic_net_A(critic_in)
                qb = self.critic_net_B(critic_in)
                return outputs_n, (qa + qb) / 2.0

            def Qtarget(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
                bs = observation.shape[0]
                outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
                critic_in = self._concat([outputs_n, actions, agent_ids])
                qa = self.target_critic_net_A(critic_in)
                qb = self.target_critic_net_B(critic_in)
                min_q = ms.ops.minimum(qa, qb)
                return outputs_n, min_q

            def Qaction_A(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
                bs = observation.shape[0]
                outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
                critic_in = self._concat([outputs_n, actions, agent_ids])
                qa = self.critic_net_A(critic_in)
                return outputs_n, qa

            def Qaction_B(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
                bs = observation.shape[0]
                outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
                critic_in = self._concat([outputs_n, actions, agent_ids])
                qb = self.critic_net_B(critic_in)
                return outputs_n, qb

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
                    tp.assign_value((tau*ep.data+(1-tau)*tp.data))
                for ep, tp in zip(self.critic_net_A.trainable_params(), self.target_critic_net_A.trainable_params()):
                    tp.assign_value((tau*ep.data+(1-tau)*tp.data))
                for ep, tp in zip(self.critic_net_B.trainable_params(), self.target_critic_net_B.trainable_params()):
                    tp.assign_value((tau*ep.data+(1-tau)*tp.data))
