Deterministic
====================================

xxxxxx.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.policies.deterministic.BasicQhead(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.torch.policies.deterministic.BasicQhead.forward(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.BasicRecurrent(**kwargs)

  :param **kwargs: xxxxxx.
  :type **kwargs: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.BasicRecurrent.forward(x, h, c)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :param h: xxxxxx.
  :type h: xxxxxx
  :param c: xxxxxx.
  :type c: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.DuelQhead(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.torch.policies.deterministic.DuelQhead.forward(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.C51Qhead(state_dim, action_dim, atom_num, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param atom_num: xxxxxx.
  :type atom_num: xxxxxx
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
  xuance.torch.policies.deterministic.C51Qhead.forward(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.QRDQNhead(state_dim, action_dim, atom_num, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param atom_num: xxxxxx.
  :type atom_num: xxxxxx
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
  xuance.torch.policies.deterministic.QRDQNhead.forward(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.BasicQnetwork(action_space, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic.BasicQnetwork.forward(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.BasicQnetwork.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.BasicQnetwork.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.DuelQnetwork(action_space, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic.DuelQnetwork.forward(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.DuelQnetwork.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.DuelQnetwork.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.NoisyQnetwork(action_space, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic.NoisyQnetwork.update_noise(noisy_bound)

  xxxxxx.

  :param noisy_bound: xxxxxx.
  :type noisy_bound: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.NoisyQnetwork.forward(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.NoisyQnetwork.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.NoisyQnetwork.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.C51Qnetwork(action_space, atom_num, vmin, vmax, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param atom_num: xxxxxx.
  :type atom_num: xxxxxx
  :param vmin: xxxxxx.
  :type vmin: xxxxxx
  :param vmax: xxxxxx.
  :type vmax: xxxxxx
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
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic.C51Qnetwork.forward(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.C51Qnetwork.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.C51Qnetwork.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.QRDQN_Network(action_space, quantile_num, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param quantile_num: xxxxxx.
  :type quantile_num: xxxxxx
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
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic.QRDQN_Network.forward(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.QRDQN_Network.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.QRDQN_Network.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.ActorNet(state_dim, action_dim, hidden_sizes, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic.ActorNet.forward(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.CriticNet(state_dim, action_dim, hidden_sizes, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic.CriticNet.forward(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.DDPGPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic.DDPGPolicy.forward(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.DDPGPolicy.Qtarget(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.DDPGPolicy.Qaction(observation, action)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.DDPGPolicy.Qpolicy(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.DDPGPolicy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.TD3Policy(action_space, representation, actor_hidden_size, critic_hidden_size, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic.TD3Policy.action(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.TD3Policy.Qtarget(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.TD3Policy.Qaction(observation, action)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.TD3Policy.Qpolicy(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.TD3Policy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.PDQNPolicy(observation_space, action_space, representation, conactor_hidden_size, qnetwork_hidden_size, normalize, initialize, activation, device)

  :param observation_space: xxxxxx.
  :type observation_space: Space
  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param conactor_hidden_size: xxxxxx.
  :type conactor_hidden_size: xxxxxx
  :param qnetwork_hidden_size: xxxxxx.
  :type qnetwork_hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic.PDQNPolicy.Atarget(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.PDQNPolicy.con_action(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.PDQNPolicy.Qtarget(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.PDQNPolicy.Qeval(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.PDQNPolicy.Qpolicy(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.PDQNPolicy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.MPDQNPolicy(observation_space, action_space, representation, conactor_hidden_size, qnetwork_hidden_size, normalize, initialize, activation, device)

  :param observation_space: xxxxxx.
  :type observation_space: Space
  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param conactor_hidden_size: xxxxxx.
  :type conactor_hidden_size: xxxxxx
  :param qnetwork_hidden_size: xxxxxx.
  :type qnetwork_hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic.MPDQNPolicy.Atarget(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.MPDQNPolicy.con_action(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.MPDQNPolicy.Qtarget(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.MPDQNPolicy.Qeval(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.MPDQNPolicy.Qpolicy(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.MPDQNPolicy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.SPDQNPolicy(observation_space, action_space, representation, conactor_hidden_size, qnetwork_hidden_size, normalize, initialize, activation, device)

  :param observation_space: xxxxxx.
  :type observation_space: Space
  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param conactor_hidden_size: xxxxxx.
  :type conactor_hidden_size: xxxxxx
  :param qnetwork_hidden_size: xxxxxx.
  :type qnetwork_hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic.SPDQNPolicy.Atarget(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.SPDQNPolicy.con_action(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.SPDQNPolicy.Qtarget(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.SPDQNPolicy.Qeval(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.SPDQNPolicy.Qpolicy(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.SPDQNPolicy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.DRQNPolicy(action_space, representation, **kwargs)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param **kwargs: xxxxxx.
  :type **kwargs: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.DRQNPolicy.forward(observation, *rnn_hidden)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :type *rnn_hidden: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.DRQNPolicy.target(observation, *rnn_hidden)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :type *rnn_hidden: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.DRQNPolicy.init_hidden(batch)

  xxxxxx.

  :param batch: xxxxxx.
  :type batch: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.DRQNPolicy.init_hidden_item(rnn_hidden, i)

  xxxxxx.

  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :param i: xxxxxx.
  :type i: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.DRQNPolicy.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.policies.deterministic.BasicQhead(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.tensorflow.policies.deterministic.BasicQhead.call(inputs)

  xxxxxx.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.BasicRecurrent(**kwargs)

  :param **kwargs: xxxxxx.
  :type **kwargs: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.BasicRecurrent.call(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.DuelQhead(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.tensorflow.policies.deterministic.DuelQhead.call(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.C51Qhead(state_dim, action_dim, atom_num, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param atom_num: xxxxxx.
  :type atom_num: xxxxxx
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
  xuance.tensorflow.policies.deterministic.C51Qhead.call(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.QRDQNhead(state_dim, action_dim, atom_num, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param atom_num: xxxxxx.
  :type atom_num: xxxxxx
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
  xuance.tensorflow.policies.deterministic.QRDQNhead.call(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.BasicQnetwork(action_space, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic.BasicQnetwork.call(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.BasicQnetwork.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.BasicQnetwork.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.DuelQnetwork(action_space, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic.DuelQnetwork.call(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.DuelQnetwork.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.DuelQnetwork.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.NoisyQnetwork(action_space, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic.NoisyQnetwork.update_noise(noisy_bound)

  xxxxxx.

  :param noisy_bound: xxxxxx.
  :type noisy_bound: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.NoisyQnetwork.call(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.NoisyQnetwork.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.NoisyQnetwork.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.C51Qnetwork(action_space, atom_num, vmin, vmax, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param atom_num: xxxxxx.
  :type atom_num: xxxxxx
  :param vmin: xxxxxx.
  :type vmin: xxxxxx
  :param vmax: xxxxxx.
  :type vmax: xxxxxx
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
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic.C51Qnetwork.call(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.C51Qnetwork.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.C51Qnetwork.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.QRDQN_Network(action_space, quantile_num, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param quantile_num: xxxxxx.
  :type quantile_num: xxxxxx
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
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic.QRDQN_Network.call(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.QRDQN_Network.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.QRDQN_Network.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.ActorNet(state_dim, action_dim, hidden_sizes, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic.ActorNet.call(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.CriticNet(state_dim, action_dim, hidden_sizes, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic.CriticNet.call(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.DDPGPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic.DDPGPolicy.call(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.DDPGPolicy.Qtarget(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.DDPGPolicy.Qaction(observation, action)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.DDPGPolicy.Qpolicy(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.DDPGPolicy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.TD3Policy(action_space, representation, actor_hidden_size, critic_hidden_size, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic.TD3Policy.action(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.TD3Policy.Qtarget(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.TD3Policy.Qaction(observation, action)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.TD3Policy.Qpolicy(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.TD3Policy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.PDQNPolicy(observation_space, action_space, representation, conactor_hidden_size, qnetwork_hidden_size, normalize, initialize, activation, device)

  :param observation_space: xxxxxx.
  :type observation_space: Space
  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param conactor_hidden_size: xxxxxx.
  :type conactor_hidden_size: xxxxxx
  :param qnetwork_hidden_size: xxxxxx.
  :type qnetwork_hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic.PDQNPolicy.Atarget(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.PDQNPolicy.con_action(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.PDQNPolicy.Qtarget(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.PDQNPolicy.Qeval(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.PDQNPolicy.Qpolicy(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.PDQNPolicy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.MPDQNPolicy(observation_space, action_space, representation, conactor_hidden_size, qnetwork_hidden_size, normalize, initialize, activation, device)

  :param observation_space: xxxxxx.
  :type observation_space: Space
  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param conactor_hidden_size: xxxxxx.
  :type conactor_hidden_size: xxxxxx
  :param qnetwork_hidden_size: xxxxxx.
  :type qnetwork_hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic.MPDQNPolicy.Atarget(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.MPDQNPolicy.con_action(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.MPDQNPolicy.Qtarget(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.MPDQNPolicy.Qeval(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.MPDQNPolicy.Qpolicy(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.MPDQNPolicy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.SPDQNPolicy(observation_space, action_space, representation, conactor_hidden_size, qnetwork_hidden_size, normalize, initialize, activation, device)

  :param observation_space: xxxxxx.
  :type observation_space: Space
  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param conactor_hidden_size: xxxxxx.
  :type conactor_hidden_size: xxxxxx
  :param qnetwork_hidden_size: xxxxxx.
  :type qnetwork_hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic.SPDQNPolicy.Atarget(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.SPDQNPolicy.con_action(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.SPDQNPolicy.Qtarget(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.SPDQNPolicy.Qeval(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.SPDQNPolicy.Qpolicy(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.SPDQNPolicy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.deterministic.DRQNPolicy(action_space, representation, **kwargs)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param **kwargs: xxxxxx.
  :type **kwargs: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.DRQNPolicy.call(observation, *rnn_hidden)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :type *rnn_hidden: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.DRQNPolicy.target(observation, *rnn_hidden)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :type *rnn_hidden: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.DRQNPolicy.init_hidden(batch)

  xxxxxx.

  :param batch: xxxxxx.
  :type batch: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.DRQNPolicy.init_hidden_item(rnn_hidden, i)

  xxxxxx.

  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :param i: xxxxxx.
  :type i: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.deterministic.DRQNPolicy.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.policies.deterministic.BasicQhead(state_dim, action_dim, hidden_sizes, normalize, initialize, activation)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.mindspore.policies.deterministic.BasicQhead.construct(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.deterministic.BasicRecurrent(kwargs)

  :param kwargs: xxxxxx.
  :type kwargs: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.BasicRecurrent.construct(x, h, c)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :param h: xxxxxx.
  :type h: xxxxxx
  :param c: xxxxxx.
  :type c: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.deterministic.DuelQhead(state_dim, action_dim, hidden_sizes, normalize, initialize, activation)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.mindspore.policies.deterministic.DuelQhead.construct(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.deterministic.C51Qhead(state_dim, action_dim, atom_num, hidden_sizes, normalize, initialize, activation)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param atom_num: xxxxxx.
  :type atom_num: xxxxxx
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic.C51Qhead.construct(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.deterministic.QRDQNhead(state_dim, action_dim, atom_num, hidden_sizes, normalize, initialize, activation)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param atom_num: xxxxxx.
  :type atom_num: xxxxxx
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic.QRDQNhead.construct(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.deterministic.BasicQnetwork(action_space, representation, hidden_sizes, normalize, initialize, activation)

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  xuance.mindspore.policies.deterministic.BasicQnetwork.construct(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.BasicQnetwork.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.BasicQnetwork.trainable_params(recurse)

  xxxxxx.

  :param recurse: xxxxxx.
  :type recurse: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.BasicQnetwork.copy_target(observation)

  xxxxxx.

.. py:class::
  xuance.mindspore.policies.deterministic.DuelQnetwork(action_space, representation, hidden_sizes, normalize, initialize, activation)

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  xuance.mindspore.policies.deterministic.DuelQnetwork.construct(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.DuelQnetwork.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.DuelQnetwork.trainable_params(recurse)

  xxxxxx.

  :param recurse: xxxxxx.
  :type recurse: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.DuelQnetwork.copy_target(observation)

  xxxxxx.

.. py:class::
  xuance.mindspore.policies.deterministic.NoisyQnetwork(action_space, representation, hidden_sizes, normalize, initialize, activation)

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  xuance.mindspore.policies.deterministic.NoisyQnetwork.update_noise(noisy_bound)

  xxxxxx.

  :param noisy_bound: xxxxxx.
  :type noisy_bound: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.NoisyQnetwork.noisy_parameters(is_target)

  xxxxxx.

  :param is_target: xxxxxx.
  :type is_target: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.NoisyQnetwork.construct(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.NoisyQnetwork.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.NoisyQnetwork.trainable_params(recurse)

  xxxxxx.

  :param recurse: xxxxxx.
  :type recurse: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.NoisyQnetwork.copy_target(observation)

  xxxxxx.

.. py:class::
  xuance.mindspore.policies.deterministic.C51Qnetwork(action_space, atom_num, vmin, vmax, representation, hidden_sizes, normalize, initialize, activation)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param atom_num: xxxxxx.
  :type atom_num: xxxxxx
  :param vmin: xxxxxx.
  :type vmin: xxxxxx
  :param vmax: xxxxxx.
  :type vmax: xxxxxx
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
  xuance.mindspore.policies.deterministic.C51Qnetwork.construct(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.C51Qnetwork.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.C51Qnetwork.copy_target(observation)

  xxxxxx.

.. py:class::
  xuance.mindspore.policies.deterministic.QRDQN_Network(action_space, quantile_num, representation, hidden_sizes, normalize, initialize, activation)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param quantile_num: xxxxxx.
  :type quantile_num: xxxxxx
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
  xuance.mindspore.policies.deterministic.QRDQN_Network.construct(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.QRDQN_Network.target(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.QRDQN_Network.trainable_params(recurse)

  xxxxxx.

  :param recurse: xxxxxx.
  :type recurse: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.DuelQnetwork.copy_target(observation)

  xxxxxx.

.. py:class::
  xuance.mindspore.policies.deterministic.ActorNet(state_dim, action_dim, hidden_sizes, initialize, activation)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic.ActorNet.construct(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.deterministic.CriticNet(state_dim, action_dim, hidden_sizes, initialize, activation)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic.CriticNet.construct(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.deterministic.DDPGPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, initialize, activation)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic.DDPGPolicy.construct(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.DDPGPolicy.Qtarget(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.DDPGPolicy.Qaction(observation, action)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.DDPGPolicy.Qpolicy(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.DDPGPolicy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. py:class::
  xuance.mindspore.policies.deterministic.TD3Policy(action_space, representation, actor_hidden_size, critic_hidden_size, initialize, activation)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic.TD3Policy.action(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.TD3Policy.Qtarget(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.TD3Policy.Qaction(observation, action)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.TD3Policy.Qpolicy(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.TD3Policy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. py:class::
  xuance.mindspore.policies.deterministic.PDQNPolicy(observation_space, action_space, representation, conactor_hidden_size, qnetwork_hidden_size, normalize, initialize, activation)

  :param observation_space: xxxxxx.
  :type observation_space: Space
  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param conactor_hidden_size: xxxxxx.
  :type conactor_hidden_size: xxxxxx
  :param qnetwork_hidden_size: xxxxxx.
  :type qnetwork_hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic.PDQNPolicy.Atarget(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.PDQNPolicy.con_action(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.PDQNPolicy.Qtarget(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.PDQNPolicy.Qeval(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.PDQNPolicy.Qpolicy(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.PDQNPolicy.construct()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.PDQNPolicy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. py:class::
  xuance.mindspore.policies.deterministic.MPDQNPolicy(observation_space, action_space, representation, conactor_hidden_size, qnetwork_hidden_size, normalize, initialize, activation)

  :param observation_space: xxxxxx.
  :type observation_space: Space
  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param conactor_hidden_size: xxxxxx.
  :type conactor_hidden_size: xxxxxx
  :param qnetwork_hidden_size: xxxxxx.
  :type qnetwork_hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic.MPDQNPolicy.Atarget(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.MPDQNPolicy.con_action(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.MPDQNPolicy.Qtarget(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.MPDQNPolicy.Qeval(state, action, input_q)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :param input_q: xxxxxx.
  :type input_q: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.MPDQNPolicy.Qpolicy(state, input_q)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param input_q: xxxxxx.
  :type input_q: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.MPDQNPolicy.construct()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.MPDQNPolicy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. py:class::
  xuance.mindspore.policies.deterministic.SPDQNPolicy(observation_space, action_space, representation, conactor_hidden_size, qnetwork_hidden_size, normalize, initialize, activation)

  :param observation_space: xxxxxx.
  :type observation_space: Space
  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param conactor_hidden_size: xxxxxx.
  :type conactor_hidden_size: xxxxxx
  :param qnetwork_hidden_size: xxxxxx.
  :type qnetwork_hidden_size: xxxxxx
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.deterministic.SPDQNPolicy.Atarget(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.SPDQNPolicy.con_action(state)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.SPDQNPolicy.Qtarget(state, action)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.SPDQNPolicy.Qeval(state, action, input_q)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :param input_q: xxxxxx.
  :type input_q: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.SPDQNPolicy.Qpolicy(state, input_q)

  xxxxxx.

  :param state: xxxxxx.
  :type state: xxxxxx
  :param input_q: xxxxxx.
  :type input_q: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.SPDQNPolicy.construct()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.SPDQNPolicy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. py:class::
  xuance.mindspore.policies.deterministic.DRQNPolicy(action_space, representation, kwargs)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param kwargs: xxxxxx.
  :type kwargs: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.DRQNPolicy.construct(observation, rnn_hidden)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.DRQNPolicy.target(observation, rnn_hidden)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.DRQNPolicy.init_hidden(batch)

  xxxxxx.

  :param batch: xxxxxx.
  :type batch: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.DRQNPolicy.init_hidden_item(rnn_hidden, i)

  xxxxxx.

  :param rnn_hidden: The final hidden state of the sequence.
  :type rnn_hidden: xxxxxx
  :param i: xxxxxx.
  :type i: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.deterministic.DRQNPolicy.copy_target()

  xxxxxx.

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        from xuance.torch.policies import *
        from xuance.torch.utils import *
        from xuance.torch.representations import Basic_Identical


        class BasicQhead(nn.Module):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(BasicQhead, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor):
                return self.model(x)


        class BasicRecurrent(nn.Module):
            def __init__(self, **kwargs):
                super(BasicRecurrent, self).__init__()
                self.lstm = False
                if kwargs["rnn"] == "GRU":
                    output = gru_block(kwargs["input_dim"],
                                       kwargs["recurrent_hidden_size"],
                                       kwargs["recurrent_layer_N"],
                                       kwargs["dropout"],
                                       kwargs["initialize"],
                                       kwargs["device"])
                elif kwargs["rnn"] == "LSTM":
                    self.lstm = True
                    output = lstm_block(kwargs["input_dim"],
                                        kwargs["recurrent_hidden_size"],
                                        kwargs["recurrent_layer_N"],
                                        kwargs["dropout"],
                                        kwargs["initialize"],
                                        kwargs["device"])
                else:
                    raise "Unknown recurrent module!"
                self.rnn_layer = output
                fc_layer = mlp_block(kwargs["recurrent_hidden_size"], kwargs["action_dim"], None, None, None, kwargs["device"])[0]
                self.model = nn.Sequential(*fc_layer)

            def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor = None):
                self.rnn_layer.flatten_parameters()
                if self.lstm:
                    output, (hn, cn) = self.rnn_layer(x, (h, c))
                    return hn, cn, self.model(output)
                else:
                    output, hn = self.rnn_layer(x, h)
                    return hn, self.model(output)


        class DuelQhead(nn.Module):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(DuelQhead, self).__init__()
                v_layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    v_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initialize, device)
                    v_layers.extend(v_mlp)
                v_layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
                a_layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    a_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initialize, device)
                    a_layers.extend(a_mlp)
                a_layers.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
                self.a_model = nn.Sequential(*a_layers)
                self.v_model = nn.Sequential(*v_layers)

            def forward(self, x: torch.Tensor):
                v = self.v_model(x)
                a = self.a_model(x)
                q = v + (a - a.mean(dim=-1).unsqueeze(dim=-1))
                return q


        class C51Qhead(nn.Module):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         atom_num: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(C51Qhead, self).__init__()
                self.action_dim = action_dim
                self.atom_num = atom_num
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim * atom_num, None, None, None, device)[0])
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor):
                dist_logits = self.model(x).view(-1, self.action_dim, self.atom_num)
                dist_probs = F.softmax(dist_logits, dim=-1)
                return dist_probs


        class QRDQNhead(nn.Module):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         atom_num: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(QRDQNhead, self).__init__()
                self.action_dim = action_dim
                self.atom_num = atom_num
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim * atom_num, None, None, None, device)[0])
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor):
                quantiles = self.model(x).view(-1, self.action_dim, self.atom_num)
                return quantiles


        class BasicQnetwork(nn.Module):
            def __init__(self,
                         action_space: Discrete,
                         representation: nn.Module,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(BasicQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                             normalize, initialize, activation, device)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)

            def forward(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                evalQ = self.eval_Qhead(outputs['state'])
                argmax_action = evalQ.argmax(dim=-1)
                return outputs, argmax_action, evalQ

            def target(self, observation: Union[np.ndarray, dict]):
                outputs_target = self.target_representation(observation)
                targetQ = self.target_Qhead(outputs_target['state'])
                argmax_action = targetQ.argmax(dim=-1)
                return outputs_target, argmax_action.detach(), targetQ.detach()

            def copy_target(self):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
                    tp.data.copy_(ep)


        class DuelQnetwork(nn.Module):
            def __init__(self,
                         action_space: Space,
                         representation: nn.Module,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(DuelQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Qhead = DuelQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                            normalize, initialize, activation, device)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)

            def forward(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                evalQ = self.eval_Qhead(outputs['state'])
                argmax_action = evalQ.argmax(dim=-1)
                return outputs, argmax_action, evalQ

            def target(self, observation: Union[np.ndarray, dict]):
                outputs = self.target_representation(observation)
                targetQ = self.target_Qhead(outputs['state'])
                argmax_action = targetQ.argmax(dim=-1)
                return outputs, argmax_action, targetQ

            def copy_target(self):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
                    tp.data.copy_(ep)


        class NoisyQnetwork(nn.Module):
            def __init__(self,
                         action_space: Discrete,
                         representation: nn.Module,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(NoisyQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                             normalize, initialize, activation, device)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)
                self.noise_scale = 0.0

            def update_noise(self, noisy_bound: float = 0.0):
                self.eval_noise_parameter = []
                self.target_noise_parameter = []
                for parameter in self.eval_Qhead.parameters():
                    self.eval_noise_parameter.append(torch.randn_like(parameter) * noisy_bound)
                    self.target_noise_parameter.append(torch.randn_like(parameter) * noisy_bound)

            def forward(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                self.update_noise(self.noise_scale)
                for parameter, noise_param in zip(self.eval_Qhead.parameters(), self.eval_noise_parameter):
                    parameter.data.copy_(parameter.data + noise_param)
                evalQ = self.eval_Qhead(outputs['state'])
                argmax_action = evalQ.argmax(dim=-1)
                return outputs, argmax_action, evalQ

            def target(self, observation: Union[np.ndarray, dict]):
                outputs = self.target_representation(observation)
                self.update_noise(self.noise_scale)
                for parameter, noise_param in zip(self.target_Qhead.parameters(), self.target_noise_parameter):
                    parameter.data.copy_(parameter.data + noise_param)
                targetQ = self.target_Qhead(outputs['state'])
                argmax_action = targetQ.argmax(dim=-1)
                return outputs, argmax_action, targetQ.detach()

            def copy_target(self):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
                    tp.data.copy_(ep)


        class C51Qnetwork(nn.Module):
            def __init__(self,
                         action_space: Discrete,
                         atom_num: int,
                         vmin: float,
                         vmax: float,
                         representation: nn.Module,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(C51Qnetwork, self).__init__()
                self.action_dim = action_space.n
                self.atom_num = atom_num
                self.vmin = vmin
                self.vmax = vmax
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Zhead = C51Qhead(self.representation.output_shapes['state'][0], self.action_dim, self.atom_num,
                                           hidden_size,
                                           normalize, initialize, activation, device)
                self.target_Zhead = copy.deepcopy(self.eval_Zhead)
                self.supports = torch.nn.Parameter(torch.linspace(self.vmin, self.vmax, self.atom_num), requires_grad=False).to(
                    device)
                self.deltaz = (vmax - vmin) / (atom_num - 1)

            def forward(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                eval_Z = self.eval_Zhead(outputs['state'])
                eval_Q = (self.supports * eval_Z).sum(-1)
                argmax_action = eval_Q.argmax(dim=-1)
                return outputs, argmax_action, eval_Z

            def target(self, observation: Union[np.ndarray, dict]):
                outputs = self.target_representation(observation)
                target_Z = self.target_Zhead(outputs['state'])
                target_Q = (self.supports * target_Z).sum(-1)
                argmax_action = target_Q.argmax(dim=-1)
                return outputs, argmax_action, target_Z

            def copy_target(self):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.eval_Zhead.parameters(), self.target_Zhead.parameters()):
                    tp.data.copy_(ep)


        class QRDQN_Network(nn.Module):
            def __init__(self,
                         action_space: Discrete,
                         quantile_num: int,
                         representation: nn.Module,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(QRDQN_Network, self).__init__()
                self.action_dim = action_space.n
                self.quantile_num = quantile_num
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Zhead = QRDQNhead(self.representation.output_shapes['state'][0], self.action_dim, self.quantile_num,
                                            hidden_size,
                                            normalize, initialize, activation, device)
                self.target_Zhead = copy.deepcopy(self.eval_Zhead)

            def forward(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                eval_Z = self.eval_Zhead(outputs['state'])
                eval_Q = eval_Z.mean(dim=-1)
                argmax_action = eval_Q.argmax(dim=-1)
                return outputs, argmax_action, eval_Z

            def target(self, observation: Union[np.ndarray, dict]):
                outputs = self.target_representation(observation)
                target_Z = self.target_Zhead(outputs['state'])
                target_Q = target_Z.mean(dim=-1)
                argmax_action = target_Q.argmax(dim=-1)
                return outputs, argmax_action, target_Z

            def copy_target(self):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.eval_Zhead.parameters(), self.target_Zhead.parameters()):
                    tp.data.copy_(ep)


        class ActorNet(nn.Module):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(ActorNet, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, nn.Tanh, initialize, device)[0])
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.tensor):
                return self.model(x)


        class CriticNet(nn.Module):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(CriticNet, self).__init__()
                layers = []
                input_shape = (state_dim + action_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.tensor, a: torch.tensor):
                return self.model(torch.concat((x, a), dim=-1))[:, 0]


        class DDPGPolicy(nn.Module):
            def __init__(self,
                         action_space: Space,
                         representation: nn.Module,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(DDPGPolicy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.representation_info_shape = representation.output_shapes
                self.representation = representation
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size, initialize,
                                      activation, device)
                self.critic = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                        initialize, activation, device)
                self.target_actor = copy.deepcopy(self.actor)
                self.target_critic = copy.deepcopy(self.critic)

            def forward(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                act = self.actor(outputs['state'])
                return outputs, act

            def Qtarget(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                act = self.target_actor(outputs['state'])
                # noise = torch.randn_like(act).clamp(-1, 1) * 0.1
                # act = (act + noise).clamp(-1, 1)
                return self.target_critic(outputs['state'], act)

            def Qaction(self, observation: Union[np.ndarray, dict], action: torch.Tensor):
                outputs = self.representation(observation)
                return self.critic(outputs['state'], action)

            def Qpolicy(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                return self.critic(outputs['state'], self.actor(outputs['state']))

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)


        class TD3Policy(nn.Module):
            def __init__(self,
                         action_space: Space,
                         representation: nn.Module,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(TD3Policy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                      initialize, activation, device)
                self.criticA = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                         initialize, activation, device)
                self.criticB = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                         initialize, activation, device)
                self.target_actor = copy.deepcopy(self.actor)
                self.target_criticA = copy.deepcopy(self.criticA)
                self.target_criticB = copy.deepcopy(self.criticB)

            def action(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                act = self.actor(outputs['state'])
                return outputs, act

            def Qtarget(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                act = self.target_actor(outputs['state'])
                noise = torch.randn_like(act).clamp(-0.1, 0.1) * 0.1
                act = (act + noise).clamp(-1, 1)
                qa = self.target_criticA(outputs['state'], act).unsqueeze(dim=1)
                qb = self.target_criticB(outputs['state'], act).unsqueeze(dim=1)
                mim_q = torch.minimum(qa, qb)
                return outputs, mim_q

            def Qaction(self, observation: Union[np.ndarray, dict], action: torch.Tensor):
                outputs = self.representation(observation)
                qa = self.criticA(outputs['state'], action).unsqueeze(dim=1)
                qb = self.criticB(outputs['state'], action).unsqueeze(dim=1)
                return outputs, torch.cat((qa, qb), axis=-1)

            def Qpolicy(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                act = self.actor(outputs['state'])
                qa = self.criticA(outputs['state'], act).unsqueeze(dim=1)
                qb = self.criticB(outputs['state'], act).unsqueeze(dim=1)
                return outputs, (qa + qb) / 2.0

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.criticA.parameters(), self.target_criticA.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.criticB.parameters(), self.target_criticB.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)


        class PDQNPolicy(nn.Module):
            def __init__(self,
                         observation_space,
                         action_space,
                         representation: nn.Module,
                         conactor_hidden_size: Sequence[int],
                         qnetwork_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(PDQNPolicy, self).__init__()
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.observation_space = observation_space
                self.action_space = action_space
                self.num_disact = self.action_space.spaces[0].n
                self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
                self.conact_size = int(self.conact_sizes.sum())

                self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                           qnetwork_hidden_size, normalize,
                                           initialize, torch.nn.modules.activation.ReLU, device)
                self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                         initialize, torch.nn.modules.activation.ReLU, device)
                self.target_conactor = copy.deepcopy(self.conactor)
                self.target_qnetwork = copy.deepcopy(self.qnetwork)

            def Atarget(self, state):
                target_conact = self.target_conactor(state)
                return target_conact

            def con_action(self, state):
                conaction = self.conactor(state)
                return conaction

            def Qtarget(self, state, action):
                input_q = torch.cat((state, action), dim=1)
                target_q = self.target_qnetwork(input_q)
                return target_q

            def Qeval(self, state, action):
                input_q = torch.cat((state, action), dim=1)
                eval_q = self.qnetwork(input_q)
                return eval_q

            def Qpolicy(self, state):
                conact = self.conactor(state)
                input_q = torch.cat((state, conact), dim=1)
                policy_q = torch.sum(self.qnetwork(input_q))
                return policy_q

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.conactor.parameters(), self.target_conactor.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.qnetwork.parameters(), self.target_qnetwork.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)


        class MPDQNPolicy(nn.Module):
            def __init__(self,
                         observation_space,
                         action_space,
                         representation: nn.Module,
                         conactor_hidden_size: Sequence[int],
                         qnetwork_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(MPDQNPolicy, self).__init__()
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.observation_space = observation_space
                self.obs_size = self.observation_space.shape[0]
                self.action_space = action_space
                self.num_disact = self.action_space.spaces[0].n
                self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
                self.conact_size = int(self.conact_sizes.sum())

                self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                           qnetwork_hidden_size, normalize,
                                           initialize, torch.nn.modules.activation.ReLU, device)
                self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                         initialize, torch.nn.modules.activation.ReLU, device)
                self.target_conactor = copy.deepcopy(self.conactor)
                self.target_qnetwork = copy.deepcopy(self.qnetwork)

                self.offsets = self.conact_sizes.cumsum()
                self.offsets = np.insert(self.offsets, 0, 0)

            def Atarget(self, state):
                target_conact = self.target_conactor(state)
                return target_conact

            def con_action(self, state):
                conaction = self.conactor(state)
                return conaction

            def Qtarget(self, state, action):
                batch_size = state.shape[0]
                Q = []
                input_q = torch.cat((state, torch.zeros_like(action)), dim=1)
                input_q = input_q.repeat(self.num_disact, 1)
                for i in range(self.num_disact):
                    input_q[i * batch_size:(i + 1) * batch_size,
                    self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                        = action[:, self.offsets[i]:self.offsets[i + 1]]
                eval_qall = self.target_qnetwork(input_q)
                for i in range(self.num_disact):
                    eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
                    if len(eval_q.shape) == 1:
                        eval_q = eval_q.unsqueeze(1)
                    Q.append(eval_q)
                Q = torch.cat(Q, dim=1)
                return Q

            def Qeval(self, state, action):
                batch_size = state.shape[0]
                Q = []
                input_q = torch.cat((state, torch.zeros_like(action)), dim=1)
                input_q = input_q.repeat(self.num_disact, 1)
                for i in range(self.num_disact):
                    input_q[i * batch_size:(i + 1) * batch_size,
                    self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                        = action[:, self.offsets[i]:self.offsets[i + 1]]
                eval_qall = self.qnetwork(input_q)
                for i in range(self.num_disact):
                    eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
                    if len(eval_q.shape) == 1:
                        eval_q = eval_q.unsqueeze(1)
                    Q.append(eval_q)
                Q = torch.cat(Q, dim=1)
                return Q

            def Qpolicy(self, state):
                conact = self.conactor(state)
                batch_size = state.shape[0]
                Q = []
                input_q = torch.cat((state, torch.zeros_like(conact)), dim=1)
                input_q = input_q.repeat(self.num_disact, 1)
                for i in range(self.num_disact):
                    input_q[i * batch_size:(i + 1) * batch_size,
                    self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                        = conact[:, self.offsets[i]:self.offsets[i + 1]]
                eval_qall = self.qnetwork(input_q)
                for i in range(self.num_disact):
                    eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
                    if len(eval_q.shape) == 1:
                        eval_q = eval_q.unsqueeze(1)
                    Q.append(eval_q)
                Q = torch.cat(Q, dim=1)
                return Q

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.conactor.parameters(), self.target_conactor.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.qnetwork.parameters(), self.target_qnetwork.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)


        class SPDQNPolicy(nn.Module):
            def __init__(self,
                         observation_space,
                         action_space,
                         representation: nn.Module,
                         conactor_hidden_size: Sequence[int],
                         qnetwork_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(SPDQNPolicy, self).__init__()
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.observation_space = observation_space
                self.action_space = action_space
                self.num_disact = self.action_space.spaces[0].n
                self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
                self.conact_size = int(self.conact_sizes.sum())
                self.qnetwork = nn.ModuleList()
                for k in range(self.num_disact):
                    self.qnetwork.append(
                        BasicQhead(self.observation_space.shape[0] + self.conact_sizes[k], 1, qnetwork_hidden_size, normalize,
                                   initialize, torch.nn.modules.activation.ReLU, device))
                self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                         initialize, torch.nn.modules.activation.ReLU, device)
                self.target_conactor = copy.deepcopy(self.conactor)
                self.target_qnetwork = copy.deepcopy(self.qnetwork)

                self.offsets = self.conact_sizes.cumsum()
                self.offsets = np.insert(self.offsets, 0, 0)

            def Atarget(self, state):
                target_conact = self.target_conactor(state)
                return target_conact

            def con_action(self, state):
                conaction = self.conactor(state)
                return conaction

            def Qtarget(self, state, action):
                target_Q = []
                for i in range(self.num_disact):
                    conact = action[:, self.offsets[i]:self.offsets[i + 1]]
                    input_q = torch.cat((state, conact), dim=1)
                    eval_q = self.target_qnetwork[i](input_q)
                    target_Q.append(eval_q)
                target_Q = torch.cat(target_Q, dim=1)
                return target_Q

            def Qeval(self, state, action):
                Q = []
                for i in range(self.num_disact):
                    conact = action[:, self.offsets[i]:self.offsets[i + 1]]
                    input_q = torch.cat((state, conact), dim=1)
                    eval_q = self.qnetwork[i](input_q)
                    Q.append(eval_q)
                Q = torch.cat(Q, dim=1)
                return Q

            def Qpolicy(self, state):
                conacts = self.conactor(state)
                Q = []
                for i in range(self.num_disact):
                    conact = conacts[:, self.offsets[i]:self.offsets[i + 1]]
                    input_q = torch.cat((state, conact), dim=1)
                    eval_q = self.qnetwork[i](input_q)
                    Q.append(eval_q)
                Q = torch.cat(Q, dim=1)
                return Q

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.conactor.parameters(), self.target_conactor.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.qnetwork.parameters(), self.target_qnetwork.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)


        class DRQNPolicy(nn.Module):
            def __init__(self,
                         action_space: Discrete,
                         representation: nn.Module,
                         **kwargs):
                super(DRQNPolicy, self).__init__()
                self.device = kwargs['device']
                self.recurrent_layer_N = kwargs['recurrent_layer_N']
                self.rnn_hidden_dim = kwargs['recurrent_hidden_size']
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                kwargs["input_dim"] = self.representation.output_shapes['state'][0]
                kwargs["action_dim"] = self.action_dim
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.cnn = True if self.representation._get_name() == "Basic_CNN" else False
                self.eval_Qhead = BasicRecurrent(**kwargs)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)

            def forward(self, observation: Union[np.ndarray, dict], *rnn_hidden: torch.Tensor):
                if self.cnn:
                    obs_shape = observation.shape
                    outputs = self.representation(observation.reshape((-1,) + obs_shape[-3:]))
                    outputs['state'] = outputs['state'].reshape(obs_shape[0:-3] + (-1,))
                else:
                    outputs = self.representation(observation)
                if self.lstm:
                    hidden_states, cell_states, evalQ = self.eval_Qhead(outputs['state'], rnn_hidden[0], rnn_hidden[1])
                else:
                    hidden_states, evalQ = self.eval_Qhead(outputs['state'], rnn_hidden[0])
                    cell_states = None
                argmax_action = evalQ[:, -1].argmax(dim=-1)
                return outputs, argmax_action, evalQ, (hidden_states, cell_states)

            def target(self, observation: Union[np.ndarray, dict], *rnn_hidden: torch.Tensor):
                if self.cnn:
                    obs_shape = observation.shape
                    outputs = self.representation(observation.reshape((-1,) + obs_shape[-3:]))
                    outputs['state'] = outputs['state'].reshape(obs_shape[0:-3] + (-1,))
                else:
                    outputs = self.representation(observation)
                if self.lstm:
                    hidden_states, cell_states, targetQ = self.target_Qhead(outputs['state'], rnn_hidden[0], rnn_hidden[1])
                else:
                    hidden_states, targetQ = self.target_Qhead(outputs['state'], rnn_hidden[0])
                    cell_states = None
                argmax_action = targetQ.argmax(dim=-1)
                return outputs, argmax_action, targetQ.detach(), (hidden_states, cell_states)

            def init_hidden(self, batch):
                hidden_states = torch.zeros(size=(self.recurrent_layer_N, batch, self.rnn_hidden_dim)).to(self.device)
                cell_states = torch.zeros_like(hidden_states).to(self.device) if self.lstm else None
                return hidden_states, cell_states

            def init_hidden_item(self, rnn_hidden, i):
                if self.lstm:
                    rnn_hidden[0][:, i] = torch.zeros(size=(self.recurrent_layer_N, self.rnn_hidden_dim)).to(self.device)
                    rnn_hidden[1][:, i] = torch.zeros(size=(self.recurrent_layer_N, self.rnn_hidden_dim)).to(self.device)
                    return rnn_hidden
                else:
                    rnn_hidden[:, i] = torch.zeros(size=(self.recurrent_layer_N, self.rnn_hidden_dim)).to(self.device)
                    return rnn_hidden

            def copy_target(self):
                for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
                    tp.data.copy_(ep)
                for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
                    tp.data.copy_(ep)


  .. group-tab:: TensorFlow

    .. code-block:: python

        import numpy as np

        from xuance.tensorflow.policies import *
        from xuance.tensorflow.utils import *
        from xuance.tensorflow.representations import Basic_Identical


        class BasicQhead(tk.Model):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(BasicQhead, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
                self.model = tk.Sequential(layers)

            def call(self, inputs: tf.Tensor, **kwargs):
                return self.model(inputs)


        class BasicRecurrent(tk.Model):
            def __init__(self, **kwargs):
                super(BasicRecurrent, self).__init__()
                self.lstm = False
                if kwargs["rnn"] == "GRU":
                    output, _ = gru_block(kwargs["input_dim"],
                                          kwargs["recurrent_hidden_size"],
                                          kwargs["recurrent_layer_N"],
                                          kwargs["dropout"],
                                          kwargs["initialize"],
                                          kwargs["device"])
                elif kwargs["rnn"] == "LSTM":
                    self.lstm = True
                    output, _ = lstm_block(kwargs["input_dim"],
                                           kwargs["recurrent_hidden_size"],
                                           kwargs["recurrent_layer_N"],
                                           kwargs["dropout"],
                                           kwargs["initialize"],
                                           kwargs["device"])
                else:
                    raise "Unknown recurrent module!"
                self.rnn_layer = output
                fc_layer = mlp_block(kwargs["recurrent_hidden_size"], kwargs["action_dim"], None, None, None, kwargs["device"])[0]
                self.model = tk.Sequential(*fc_layer)

            def call(self, x: tf.Tensor, **kwargs):
                if self.lstm:
                    output, hn, cn = self.rnn_layer(x)
                    return hn, cn, self.model(output)
                else:
                    output, hn = self.rnn_layer(x)
                    return hn, self.model(output)


        class DuelQhead(tk.Model):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(DuelQhead, self).__init__()
                v_layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    v_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initializer, device)
                    v_layers.extend(v_mlp)
                v_layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
                a_layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    a_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initializer, device)
                    a_layers.extend(a_mlp)
                a_layers.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
                self.a_model = tk.Sequential(a_layers)
                self.v_model = tk.Sequential(v_layers)

            def call(self, x: tf.Tensor, **kwargs):
                v = self.v_model(x)
                a = self.a_model(x)
                q = v + (a - tf.expand_dims(tf.reduce_mean(a, axis=-1), axis=-1))
                return q


        class C51Qhead(tk.Model):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         atom_num: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(C51Qhead, self).__init__()
                self.action_dim = action_dim
                self.atom_num = atom_num
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim * atom_num, None, None, None, device)[0])
                self.model = tk.Sequential(layers)

            def call(self, x: tf.Tensor, **kwargs):
                dist_logits = tf.reshape(self.model(x), [-1, self.action_dim, self.atom_num])
                dist_probs = tf.nn.softmax(dist_logits, axis=-1)
                return dist_probs


        class QRDQNhead(tk.Model):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         atom_num: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(QRDQNhead, self).__init__()
                self.action_dim = action_dim
                self.atom_num = atom_num
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim * atom_num, None, None, None, device)[0])
                self.model = tk.Sequential(layers)

            def call(self, x: tf.Tensor, **kwargs):
                quantiles = tf.reshape(self.model(x), [-1, self.action_dim, self.atom_num])
                return quantiles


        class BasicQnetwork(tk.Model):
            def __init__(self,
                         action_space: Discrete,
                         representation: tk.Model,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(BasicQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                             normalize, initializer, activation, device)
                self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                               normalize, initializer, activation, device)
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

            def call(self, observation: tf.Tensor, **kwargs):
                outputs = self.representation(observation)
                evalQ = self.eval_Qhead(outputs['state'])
                argmax_action = tf.math.argmax(evalQ, axis=-1)
                return outputs, argmax_action, evalQ

            def target(self, observation: Union[np.ndarray, dict]):
                outputs_target = self.target_representation(observation)
                targetQ = self.target_Qhead(outputs_target['state'])
                argmax_action = tf.math.argmax(targetQ, axis=-1)
                return outputs_target, tf.stop_gradient(argmax_action), tf.stop_gradient(targetQ)

            def copy_target(self):
                self.target_representation.set_weights(self.representation.get_weights())
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


        class DuelQnetwork(tk.Model):
            def __init__(self,
                         action_space: Space,
                         representation: Basic_Identical,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[tk.layers.Layer] = None,
                         initialize: Optional[Callable[..., tf.Tensor]] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(DuelQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Qhead = DuelQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                            normalize, initialize, activation, device)
                self.target_Qhead = DuelQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                              normalize, initialize, activation, device)
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

            def call(self, observation: Union[np.ndarray, dict], **kwargs):
                outputs = self.representation(observation)
                evalQ = self.eval_Qhead(outputs['state'])
                argmax_action = tf.math.argmax(evalQ, axis=-1)
                return outputs, argmax_action, evalQ

            def target(self, observation: Union[np.ndarray, dict]):
                outputs = self.target_representation(observation)
                targetQ = self.target_Qhead(outputs['state'])
                argmax_action = tf.math.argmax(targetQ, axis=-1)
                return outputs, argmax_action, targetQ

            def copy_target(self):
                self.target_representation.set_weights(self.representation.get_weights())
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


        class NoisyQnetwork(tk.Model):
            def __init__(self,
                         action_space: Discrete,
                         representation: Basic_Identical,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[tk.layers.Layer] = None,
                         initialize: Optional[Callable[..., tf.Tensor]] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(NoisyQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                             normalize, initialize, activation, device)
                self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                               normalize, initialize, activation, device)
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
                self.noise_scale = 0.0

            def update_noise(self, noisy_bound: float = 0.0):
                self.eval_noise_parameter = []
                self.target_noise_parameter = []
                for parameter in self.eval_Qhead.variables:
                    self.eval_noise_parameter.append(
                        tf.random.uniform(shape=parameter.shape, minval=-1.0, maxval=1.0) * noisy_bound)
                    self.target_noise_parameter.append(
                        tf.random.uniform(shape=parameter.shape, minval=-1.0, maxval=1.0) * noisy_bound)

            def call(self, observation: Union[np.ndarray, dict], **kwargs):
                outputs = self.representation(observation)
                self.update_noise(self.noise_scale)
                for parameter, noise_param in zip(self.eval_Qhead.variables, self.eval_noise_parameter):
                    parameter.assign_add(noise_param)
                evalQ = self.eval_Qhead(outputs['state'])
                argmax_action = tf.math.argmax(evalQ, axis=-1)
                return outputs, argmax_action, evalQ

            def target(self, observation: Union[np.ndarray, dict]):
                outputs = self.target_representation(observation)
                self.update_noise(self.noise_scale)
                for parameter, noise_param in zip(self.target_Qhead.variables, self.target_noise_parameter):
                    parameter.assign_add(noise_param)
                targetQ = self.target_Qhead(outputs['state'])
                argmax_action = tf.math.argmax(targetQ, axis=-1)
                return outputs, argmax_action, tf.stop_gradient(targetQ)

            def copy_target(self):
                self.target_representation.set_weights(self.representation.get_weights())
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


        class C51Qnetwork(tk.Model):
            def __init__(self,
                         action_space: Discrete,
                         atom_num: int,
                         vmin: float,
                         vmax: float,
                         representation: Basic_Identical,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[tk.layers.Layer] = None,
                         initialize: Optional[Callable[..., tf.Tensor]] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                assert isinstance(action_space, Discrete)
                super(C51Qnetwork, self).__init__()
                self.action_dim = action_space.n
                self.atom_num = atom_num
                self.vmin = vmin
                self.vmax = vmax
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Zhead = C51Qhead(self.representation.output_shapes['state'][0], self.action_dim, self.atom_num,
                                           hidden_size, normalize, initialize, activation, device)
                self.target_Zhead = C51Qhead(self.representation.output_shapes['state'][0], self.action_dim, self.atom_num,
                                             hidden_size, normalize, initialize, activation, device)
                self.target_Zhead.set_weights(self.eval_Zhead.get_weights())
                self.supports = tf.cast(tf.linspace(self.vmin, self.vmax, self.atom_num), dtype=tf.float32)
                self.deltaz = (vmax - vmin) / (atom_num - 1)

            def call(self, observation: Union[np.ndarray, dict], **kwargs):
                outputs = self.representation(observation)
                eval_Z = self.eval_Zhead(outputs['state'])
                eval_Q = tf.reduce_sum(self.supports * eval_Z, axis=-1)
                argmax_action = tf.math.argmax(eval_Q, axis=-1)
                return outputs, argmax_action, eval_Z

            def target(self, observation: Union[np.ndarray, dict]):
                outputs = self.target_representation(observation)
                target_Z = self.target_Zhead(outputs['state'])
                target_Q = tf.reduce_sum(self.supports * target_Z, axis=-1)
                argmax_action = tf.math.argmax(target_Q, axis=-1)
                return outputs, argmax_action, target_Z

            def copy_target(self):
                self.target_representation.set_weights(self.representation.get_weights())
                self.target_Zhead.set_weights(self.eval_Zhead.get_weights())


        class QRDQN_Network(tk.Model):
            def __init__(self,
                         action_space: Discrete,
                         quantile_num: int,
                         representation: Basic_Identical,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[tk.layers.Layer] = None,
                         initialize: Optional[Callable[..., tf.Tensor]] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(QRDQN_Network, self).__init__()
                self.action_dim = action_space.n
                self.quantile_num = quantile_num
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Zhead = QRDQNhead(self.representation.output_shapes['state'][0], self.action_dim, self.quantile_num,
                                            hidden_size,
                                            normalize, initialize, activation, device)
                self.target_Zhead = QRDQNhead(self.representation.output_shapes['state'][0], self.action_dim, self.quantile_num,
                                              hidden_size,
                                              normalize, initialize, activation, device)
                self.target_Zhead.set_weights(self.eval_Zhead.get_weights())

            def call(self, observation: Union[np.ndarray, dict], **kwargs):
                outputs = self.representation(observation)
                eval_Z = self.eval_Zhead(outputs['state'])
                eval_Q = tf.reduce_mean(eval_Z, axis=-1)
                argmax_action = tf.math.argmax(eval_Q, axis=-1)
                return outputs, argmax_action, eval_Z

            def target(self, observation: Union[np.ndarray, dict]):
                outputs = self.target_representation(observation)
                target_Z = self.target_Zhead(outputs['state'])
                target_Q = tf.reduce_mean(target_Z, axis=-1)
                argmax_action = tf.math.argmax(target_Q, axis=-1)
                return outputs, argmax_action, target_Z

            def copy_target(self):
                self.target_representation.set_weights(self.representation.get_weights())
                self.target_Zhead.set_weights(self.eval_Zhead.get_weights())


        class ActorNet(tk.Model):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         initialize: Optional[Callable[..., tf.Tensor]] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(ActorNet, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, tk.layers.Activation('tanh'), initialize, device)[0])
                self.model = tk.Sequential(layers)

            def call(self, x: tf.Tensor, **kwargs):
                return self.model(x)


        class CriticNet(tk.Model):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         initialize: Optional[Callable[..., tf.Tensor]] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(CriticNet, self).__init__()
                layers = []
                input_shape = (state_dim + action_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
                self.model = tk.Sequential(layers)

            def call(self, inputs: Dict, **kwargs):
                x = inputs['x']
                a = inputs['a']
                return self.model(tf.concat((x, a), axis=-1))[:, 0]


        class DDPGPolicy(tk.Model):
            def __init__(self,
                         action_space: Space,
                         representation: Basic_Identical,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         initialize: Optional[Callable[..., tf.Tensor]] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(DDPGPolicy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes

                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size, initialize,
                                      activation, device)
                self.critic = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                        initialize, activation, device)
                self.target_actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                             initialize,
                                             activation, device)
                self.target_critic = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                               initialize, activation, device)
                self.soft_update(tau=1.0)

            def call(self, observation: Union[np.ndarray, dict], **kwargs):
                outputs = self.representation(observation)
                act = self.actor(outputs['state'])
                return outputs, act

            def Qtarget(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                act = self.target_actor(outputs['state'])
                inputs_critic = {'x': outputs['state'], 'a': act}
                return self.target_critic(inputs_critic)

            def Qaction(self, observation: Union[np.ndarray, dict], action: tf.Tensor):
                outputs = self.representation(observation)
                inputs_critic = {'x': outputs['state'], 'a': action}
                return self.critic(inputs_critic)

            def Qpolicy(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                action = self.actor(outputs['state'])
                inputs_critic = {'x': outputs['state'], 'a': action}
                return self.critic(inputs_critic)

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor.variables, self.target_actor.variables):
                    tp.assign((1 - tau) * tp + tau * ep)
                for ep, tp in zip(self.critic.variables, self.target_critic.variables):
                    tp.assign((1 - tau) * tp + tau * ep)


        class TD3Policy(tk.Model):
            def __init__(self,
                         action_space: Space,
                         representation: Basic_Identical,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initialize: Optional[Callable[..., tf.Tensor]] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(TD3Policy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                      initialize, activation, device)
                self.criticA = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                         initialize, activation, device)
                self.criticB = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                         initialize, activation, device)
                self.target_actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                             initialize, activation, device)
                self.target_criticA = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                                initialize, activation, device)
                self.target_criticB = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                                initialize, activation, device)
                self.target_criticA.set_weights(self.criticA.get_weights())
                self.target_criticB.set_weights(self.criticB.get_weights())

            def call(self, observation: Union[np.ndarray, dict], **kwargs):
                outputs = self.representation(observation)
                act = self.actor(outputs['state'])
                return outputs, act

            def Qtarget(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                act = self.target_actor(outputs['state'])
                noise = tf.random.uniform(act.shape, -1, 1) * 0.1
                act = tf.clip_by_value(act + noise, -1, 1)
                inputs_critic = {'x': outputs['state'], 'a': act}
                qa = tf.expand_dims(self.target_criticA(inputs_critic), axis=1)
                qb = tf.expand_dims(self.target_criticB(inputs_critic), axis=1)
                mim_q = tf.minimum(qa, qb)
                return outputs, mim_q

            def Qaction(self, observation: Union[np.ndarray, dict], action: tf.Tensor):
                outputs = self.representation(observation)
                inputs_critic = {'x': outputs['state'], 'a': action}
                qa = tf.expand_dims(self.criticA(inputs_critic), axis=1)
                qb = tf.expand_dims(self.criticB(inputs_critic), axis=1)
                return outputs, tf.concat((qa, qb), axis=-1)

            def Qpolicy(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                act = self.actor(outputs['state'])
                inputs_critic = {'x': outputs['state'], 'a': act}
                qa = tf.expand_dims(self.criticA(inputs_critic), axis=1)
                qb = tf.expand_dims(self.criticB(inputs_critic), axis=1)
                return outputs, (qa + qb) / 2.0

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor.variables, self.target_actor.variables):
                    tp.assign((1 - tau) * tp + tau * ep)
                for ep, tp in zip(self.criticA.variables, self.target_criticA.variables):
                    tp.assign((1 - tau) * tp + tau * ep)
                for ep, tp in zip(self.criticB.variables, self.target_criticB.variables):
                    tp.assign((1 - tau) * tp + tau * ep)


        class PDQNPolicy(tk.Model):
            def __init__(self,
                         observation_space,
                         action_space,
                         representation: Basic_Identical,
                         conactor_hidden_size: Sequence[int],
                         qnetwork_hidden_size: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initialize: Optional[Callable[..., tf.Tensor]] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(PDQNPolicy, self).__init__()
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.observation_space = observation_space
                self.action_space = action_space
                self.num_disact = self.action_space.spaces[0].n
                self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
                self.conact_size = int(self.conact_sizes.sum())

                self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                           qnetwork_hidden_size, normalize,
                                           initialize, activation, device)
                self.dim_input = self.observation_space.shape[0] + self.conact_size
                self.qnetwork._set_inputs(tf.TensorSpec([None, self.dim_input], tf.float32, name='inputs'))
                self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                         initialize, activation, device)
                self.target_qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                                  qnetwork_hidden_size, normalize, initialize, activation, device)
                self.target_conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                                initialize, activation, device)
                self.target_conactor.set_weights(self.conactor.get_weights())
                self.target_qnetwork.set_weights(self.qnetwork.get_weights())

            def Atarget(self, state):
                target_conact = self.target_conactor(state)
                return target_conact

            def con_action(self, state):
                conaction = self.conactor(state)
                return conaction

            def Qtarget(self, state, action):
                input_q = tf.concat((state, action), axis=1)
                target_q = self.target_qnetwork(input_q)
                return target_q

            def Qeval(self, state, action):
                input_q = tf.concat((state, action), axis=1)
                eval_q = self.qnetwork(input_q)
                return eval_q

            def Qpolicy(self, state):
                conact = self.conactor(state)
                input_q = tf.concat((state, conact), axis=1)
                policy_q = tf.reduce_sum(self.qnetwork(input_q))
                return policy_q

            def soft_update(self, tau=0.005):
                # for ep, tp in zip(self.representation.variables, self.target_representation.variables):
                #     tp.assign((1 - tau) * tp + tau * ep)
                for ep, tp in zip(self.conactor.variables, self.target_conactor.variables):
                    tp.assign((1 - tau) * tp + tau * ep)
                for ep, tp in zip(self.qnetwork.variables, self.target_qnetwork.variables):
                    tp.assign((1 - tau) * tp + tau * ep)


        class MPDQNPolicy(tk.Model):
            def __init__(self,
                         observation_space,
                         action_space,
                         representation: Basic_Identical,
                         conactor_hidden_size: Sequence[int],
                         qnetwork_hidden_size: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initialize: Optional[Callable[..., tf.Tensor]] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(MPDQNPolicy, self).__init__()
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.observation_space = observation_space
                self.obs_size = self.observation_space.shape[0]
                self.action_space = action_space
                self.num_disact = self.action_space.spaces[0].n
                self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
                self.conact_size = int(self.conact_sizes.sum())

                self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                           qnetwork_hidden_size, normalize,
                                           initialize, activation, device)
                self.dim_input = self.observation_space.shape[0] + self.conact_size
                self.qnetwork._set_inputs(tf.TensorSpec([None, self.dim_input], tf.float32, name='inputs'))
                self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                         initialize, activation, device)

                self.target_qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                                  qnetwork_hidden_size, normalize,
                                                  initialize, activation, device)
                self.target_conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                                initialize, activation, device)
                self.offsets = self.conact_sizes.cumsum()
                self.offsets = np.insert(self.offsets, 0, 0)
                self.soft_update(tau=1.0)

            def Atarget(self, state):
                target_conact = self.target_conactor(state)
                return target_conact

            def con_action(self, state):
                conaction = self.conactor(state)
                return conaction

            def Qtarget(self, state, action):
                batch_size = state.shape[0]
                Q = []
                input_q = tf.concat((state, tf.zeros_like(action)), axis=1)
                input_q = tf.tile(input_q, (self.num_disact, 1)).numpy()
                for i in range(self.num_disact):
                    input_q[i * batch_size:(i + 1) * batch_size,
                    self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                        = action[:, self.offsets[i]:self.offsets[i + 1]]
                eval_qall = self.target_qnetwork(input_q)
                for i in range(self.num_disact):
                    eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
                    if len(eval_q.shape) == 1:
                        eval_q = tf.expand_dims(eval_q, 1)
                    Q.append(eval_q)
                Q = tf.concat(Q, axis=1)
                return Q

            def Qeval(self, state, action):
                batch_size = state.shape[0]
                Q = []
                input_q = tf.concat((state, tf.zeros_like(action)), axis=1)
                input_q = tf.tile(input_q, (self.num_disact, 1)).numpy()
                for i in range(self.num_disact):
                    input_q[i * batch_size:(i + 1) * batch_size,
                    self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                        = action[:, self.offsets[i]:self.offsets[i + 1]]
                eval_qall = self.qnetwork(input_q)
                for i in range(self.num_disact):
                    eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
                    if len(eval_q.shape) == 1:
                        eval_q = tf.expand_dims(eval_q, axis=1)
                    Q.append(eval_q)
                Q = tf.concat(Q, axis=1)
                return Q

            def Qpolicy(self, state):
                conact = self.conactor(state)
                batch_size = state.shape[0]
                Q = []
                input_q = tf.concat((state, tf.zeros_like(conact)), axis=1)
                input_q = tf.tile(input_q, (self.num_disact, 1)).numpy()
                for i in range(self.num_disact):
                    input_q[i * batch_size:(i + 1) * batch_size,
                    self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                        = conact[:, self.offsets[i]:self.offsets[i + 1]]
                eval_qall = self.qnetwork(input_q)
                for i in range(self.num_disact):
                    eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
                    if len(eval_q.shape) == 1:
                        eval_q = tf.expand_dims(eval_q, axis=1)
                    Q.append(eval_q)
                Q = tf.concat(Q, axis=1)
                return Q

            def soft_update(self, tau=0.005):
                # for ep, tp in zip(self.representation.variables, self.target_representation.variables):
                #     tp.assign((1 - tau) * tp + tau * ep)
                for ep, tp in zip(self.conactor.variables, self.target_conactor.variables):
                    tp.assign((1 - tau) * tp + tau * ep)
                for ep, tp in zip(self.qnetwork.variables, self.target_qnetwork.variables):
                    tp.assign((1 - tau) * tp + tau * ep)


        class SPDQNPolicy(tk.Model):
            def __init__(self,
                         observation_space,
                         action_space,
                         representation: Basic_Identical,
                         conactor_hidden_size: Sequence[int],
                         qnetwork_hidden_size: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initialize: Optional[Callable[..., tf.Tensor]] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(SPDQNPolicy, self).__init__()
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.observation_space = observation_space
                self.action_space = action_space
                self.num_disact = self.action_space.spaces[0].n
                self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
                self.conact_size = int(self.conact_sizes.sum())
                self.qnetwork, self.target_qnetwork = [], []
                for k in range(self.num_disact):
                    self.qnetwork.append(
                        BasicQhead(self.observation_space.shape[0] + self.conact_sizes[k], 1, qnetwork_hidden_size, normalize,
                                   initialize, activation, device))
                    dim_input = self.observation_space.shape[0] + self.conact_sizes[k]
                    self.qnetwork[k]._set_inputs(tf.TensorSpec([None, dim_input], tf.float32, name='inputs_%d'%(k)))

                self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                         initialize, activation, device)
                for k in range(self.num_disact):
                    self.target_qnetwork.append(
                        BasicQhead(self.observation_space.shape[0] + self.conact_sizes[k], 1, qnetwork_hidden_size, normalize,
                                   initialize, activation, device))
                self.target_conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                                initialize, activation, device)

                self.offsets = self.conact_sizes.cumsum()
                self.offsets = np.insert(self.offsets, 0, 0)
                self.soft_update(tau=1.0)

            def Atarget(self, state):
                target_conact = self.target_conactor(state)
                return target_conact

            def con_action(self, state):
                conaction = self.conactor(state)
                return conaction

            def Qtarget(self, state, action):
                target_Q = []
                for i in range(self.num_disact):
                    conact = action[:, self.offsets[i]:self.offsets[i + 1]]
                    input_q = tf.concat((state, conact), axis=1)
                    eval_q = self.target_qnetwork[i](input_q)
                    target_Q.append(eval_q)
                target_Q = tf.concat(target_Q, axis=1)
                return target_Q

            def Qeval(self, state, action):
                Q = []
                for i in range(self.num_disact):
                    conact = action[:, self.offsets[i]:self.offsets[i + 1]]
                    input_q = tf.concat((state, conact), axis=1)
                    eval_q = self.qnetwork[i](input_q)
                    Q.append(eval_q)
                Q = tf.concat(Q, axis=1)
                return Q

            def Qpolicy(self, state):
                conacts = self.conactor(state)
                Q = []
                for i in range(self.num_disact):
                    conact = conacts[:, self.offsets[i]:self.offsets[i + 1]]
                    input_q = tf.concat((state, conact), axis=1)
                    eval_q = self.qnetwork[i](input_q)
                    Q.append(eval_q)
                Q = tf.concat(Q, axis=1)
                return Q

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.conactor.variables, self.target_conactor.variables):
                    tp.assign((1 - tau) * tp + tau * ep)
                for qnet, target_qnet in zip(self.qnetwork, self.target_qnetwork):
                    for ep, tp in zip(qnet.variables, target_qnet.variables):
                        tp.assign((1 - tau) * tp + tau * ep)


        class DRQNPolicy(tk.Model):
            def __init__(self,
                         action_space: Discrete,
                         representation: tk.Model,
                         **kwargs):
                super(DRQNPolicy, self).__init__()
                self.device = kwargs['device']
                self.recurrent_layer_N = kwargs['recurrent_layer_N']
                self.rnn_hidden_dim = kwargs['recurrent_hidden_size']
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                kwargs["input_dim"] = self.representation.output_shapes['state'][0]
                kwargs["action_dim"] = self.action_dim
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.cnn = True if self.representation.name == "basic_cnn" else False
                self.eval_Qhead = BasicRecurrent(**kwargs)
                self.target_Qhead = BasicRecurrent(**kwargs)
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

            def call(self, observation: Union[np.ndarray, dict], *rnn_hidden: tf.Tensor, **kwargs):
                if self.cnn:
                    obs_shape = observation.shape
                    outputs = self.representation(observation.reshape((-1,) + obs_shape[-3:]))
                    output_states = outputs['state'].reshape(obs_shape[0:-3] + (-1,))
                else:
                    obs_shape = observation.shape
                    observations_in = tf.reshape(observation, [-1, obs_shape[-1]])
                    outputs = self.representation(observations_in)
                    output_states = tf.reshape(outputs['state'], obs_shape[:-1] + self.representation.output_shapes['state'])
                if self.lstm:
                    hidden_states, cell_states, evalQ = self.eval_Qhead(output_states)
                else:
                    hidden_states, evalQ = self.eval_Qhead(output_states)
                    cell_states = None
                argmax_action = tf.math.argmax(evalQ[:, -1], axis=-1)
                return outputs, argmax_action, evalQ, (hidden_states, cell_states)

            def target(self, observation: Union[np.ndarray, dict], *rnn_hidden: tf.Tensor):
                if self.cnn:
                    obs_shape = observation.shape
                    outputs = self.target_representation(observation.reshape((-1,) + obs_shape[-3:]))
                    output_states = outputs['state'].reshape(obs_shape[0:-3] + (-1,))
                else:
                    obs_shape = observation.shape
                    observations_in = tf.reshape(observation, [-1, obs_shape[-1]])
                    outputs = self.target_representation(observations_in)
                    output_states = tf.reshape(outputs['state'], obs_shape[:-1] + self.representation.output_shapes['state'])
                if self.lstm:
                    hidden_states, cell_states, targetQ = self.target_Qhead(output_states)
                else:
                    hidden_states, targetQ = self.target_Qhead(output_states)
                    cell_states = None
                argmax_action = tf.math.argmax(targetQ, axis=-1)
                return outputs, argmax_action, targetQ, (hidden_states, cell_states)

            def init_hidden(self, batch):
                with tf.device(self.device):
                    hidden_states = tf.zeros(shape=(self.recurrent_layer_N, batch, self.rnn_hidden_dim))
                    cell_states = tf.zeros_like(hidden_states) if self.lstm else None
                    return hidden_states, cell_states

            def init_hidden_item(self, rnn_hidden, i):
                with tf.device(self.device):
                    if self.lstm:
                        rnn_hidden_0, rnn_hidden_1 = rnn_hidden[0].numpy(), rnn_hidden[1].numpy()
                        rnn_hidden_0[i:i+1] = np.zeros((self.recurrent_layer_N, self.rnn_hidden_dim))
                        rnn_hidden_1[i:i+1] = np.zeros((self.recurrent_layer_N, self.rnn_hidden_dim))
                        return (tf.convert_to_tensor(rnn_hidden_0), tf.convert_to_tensor(rnn_hidden_1))
                    else:
                        rnn_hidden_np = rnn_hidden.numpy()
                        rnn_hidden_np[i:i+1] = np.zeros((self.recurrent_layer_N, self.rnn_hidden_dim))
                        return tf.convert_to_tensor(rnn_hidden_np)

            def copy_target(self):
                self.target_representation.set_weights(self.representation.get_weights())
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())



  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.policies import *
        from xuance.mindspore.utils import *
        import copy
        from gym.spaces import Space, Box, Discrete, Dict


        class BasicQhead(nn.Cell):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(BasicQhead, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, None, None)[0])
                self.model = nn.SequentialCell(*layers)

            def construct(self, x: ms.tensor):
                return self.model(x)


        class BasicRecurrent(nn.Cell):
            def __init__(self, **kwargs):
                super(BasicRecurrent, self).__init__()
                self.lstm = False
                if kwargs["rnn"] == "GRU":
                    output, _ = gru_block(kwargs["input_dim"],
                                          kwargs["recurrent_hidden_size"],
                                          kwargs["recurrent_layer_N"],
                                          kwargs["dropout"],
                                          kwargs["initialize"])
                elif kwargs["rnn"] == "LSTM":
                    self.lstm = True
                    output, _ = lstm_block(kwargs["input_dim"],
                                           kwargs["recurrent_hidden_size"],
                                           kwargs["recurrent_layer_N"],
                                           kwargs["dropout"],
                                           kwargs["initialize"])
                else:
                    raise "Unknown recurrent module!"
                self.rnn_layer = output
                fc_layer = mlp_block(kwargs["recurrent_hidden_size"], kwargs["action_dim"], None, None, None)[0]
                self.model = nn.SequentialCell(*fc_layer)

            def construct(self, x: ms.tensor, h: ms.tensor, c: ms.tensor = None):
                # self.rnn_layer.flatten_parameters()
                if self.lstm:
                    output, (hn, cn) = self.rnn_layer(x, (h, c))
                    return hn, cn, self.model(output)
                else:
                    output, hn = self.rnn_layer(x, h)
                    return hn, self.model(output)


        class DuelQhead(nn.Cell):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(DuelQhead, self).__init__()
                v_layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    v_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initialize)
                    v_layers.extend(v_mlp)
                v_layers.extend(mlp_block(input_shape[0], 1, None, None, None)[0])

                a_layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    a_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initialize)
                    a_layers.extend(a_mlp)
                a_layers.extend(mlp_block(input_shape[0], action_dim, None, None, None)[0])

                self.a_model = nn.SequentialCell(*a_layers)
                self.v_model = nn.SequentialCell(*v_layers)

                self._mean = ms.ops.ReduceMean(keep_dims=True)

            def construct(self, x: ms.tensor):
                v = self.v_model(x)
                a = self.a_model(x)
                q = v + (a - self._mean(a))
                return q


        class C51Qhead(nn.Cell):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         atom_num: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(C51Qhead, self).__init__()
                self.action_dim = action_dim
                self.atom_num = atom_num
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim * atom_num, None, None, None)[0])
                self.model = nn.SequentialCell(*layers)
                self._softmax = ms.ops.Softmax(axis=-1)

            def construct(self, x: ms.tensor):
                dist_logits = self.model(x).view(-1, self.action_dim, self.atom_num)
                dist_probs = self._softmax(dist_logits)
                return dist_probs


        class QRDQNhead(nn.Cell):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         atom_num: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(QRDQNhead, self).__init__()
                self.action_dim = action_dim
                self.atom_num = atom_num
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim * atom_num, None, None, None)[0])
                self.model = nn.SequentialCell(*layers)

            def construct(self, x: ms.tensor):
                return self.model(x).view(-1, self.action_dim, self.atom_num)


        class BasicQnetwork(nn.Cell):
            def __init__(self,
                         action_space: Discrete,
                         representation: ModuleType,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(BasicQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                             normalize, initialize, activation)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)

            def construct(self, observation: ms.tensor):
                outputs = self.representation(observation)
                evalQ = self.eval_Qhead(outputs['state'])
                argmax_action = evalQ.argmax(axis=-1)
                return outputs, argmax_action, evalQ

            def target(self, observation: ms.tensor):
                outputs_target = self.target_representation(observation)
                targetQ = self.target_Qhead(outputs_target['state'])
                argmax_action = targetQ.argmax(axis=-1)
                return outputs_target, argmax_action, targetQ

            def trainable_params(self, recurse=True):
                return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

            def copy_target(self):
                for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
                    tp.assign_value(ep)


        class DuelQnetwork(nn.Cell):
            def __init__(self,
                         action_space: Space,
                         representation: ModuleType,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(DuelQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Qhead = DuelQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                            normalize, initialize, activation)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)

            def construct(self, observation: ms.tensor):
                outputs = self.representation(observation)
                evalQ = self.eval_Qhead(outputs['state'])
                argmax_action = evalQ.argmax(axis=-1)
                return outputs, argmax_action, evalQ

            def target(self, observation: ms.tensor):
                outputs = self.target_representation(observation)
                targetQ = self.target_Qhead(outputs['state'])
                argmax_action = targetQ.argmax(axis=-1)
                return outputs, argmax_action, targetQ

            def trainable_params(self, recurse=True):
                return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

            def copy_target(self):
                for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
                    tp.assign_value(ep)


        class NoisyQnetwork(nn.Cell):
            def __init__(self,
                         action_space: Discrete,
                         representation: ModuleType,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(NoisyQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                             normalize, initialize, activation)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)

                self._stdnormal = ms.ops.StandardNormal()
                self._assign = ms.ops.Assign()

            def update_noise(self, noisy_bound: float = 0.0):
                self.eval_noise_parameter = []
                self.target_noise_parameter = []
                for parameter in self.eval_Qhead.trainable_params():
                    self.eval_noise_parameter.append(self._stdnormal(parameter.shape) * noisy_bound)
                    self.target_noise_parameter.append(self._stdnormal(parameter.shape) * noisy_bound)

            def noisy_parameters(self, is_target=False):
                self.update_noise(self.noise_scale)
                if is_target:
                    for parameter, noise_param in zip(self.eval_Qhead.trainable_params(), self.eval_noise_parameter):
                        _ = self._assign(parameter, parameter + noise_param)
                else:
                    for parameter, noise_param in zip(self.target_Qhead.trainable_params(), self.target_noise_parameter):
                        _ = self._assign(parameter, parameter + noise_param)

            def construct(self, observation: ms.tensor):
                outputs = self.representation(observation)
                evalQ = self.eval_Qhead(outputs['state'])
                argmax_action = evalQ.argmax(axis=-1)
                return outputs, argmax_action, evalQ

            def target(self, observation: ms.tensor):
                outputs = self.target_representation(observation)
                self.noisy_parameters(is_target=True)
                targetQ = self.target_Qhead(outputs['state'])
                argmax_action = targetQ.argmax(axis=-1)
                return outputs, argmax_action, targetQ

            def trainable_params(self, recurse=True):
                return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

            def copy_target(self):
                for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
                    tp.assign_value(ep)


        class C51Qnetwork(nn.Cell):
            def __init__(self,
                         action_space: Discrete,
                         atom_num: int,
                         vmin: float,
                         vmax: float,
                         representation: ModuleType,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                assert isinstance(action_space, Discrete)
                super(C51Qnetwork, self).__init__()
                self.action_dim = action_space.n
                self.atom_num = atom_num
                self.vmin = vmin
                self.vmax = vmax
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Zhead = C51Qhead(self.representation.output_shapes['state'][0], self.action_dim, self.atom_num,
                                           hidden_size, normalize, initialize, activation)
                self.target_Zhead = copy.deepcopy(self.eval_Zhead)
                self._LinSpace = ms.ops.LinSpace()
                self.supports = ms.Parameter(self._LinSpace(ms.Tensor(self.vmin, ms.float32),
                                                            ms.Tensor(self.vmax, ms.float32),
                                                            self.atom_num),
                                             requires_grad=False)
                self.deltaz = (vmax - vmin) / (atom_num - 1)

            def construct(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                eval_Z = self.eval_Zhead(outputs['state'])
                eval_Q = (self.supports * eval_Z).sum(-1)
                argmax_action = eval_Q.argmax(axis=-1)
                return outputs, argmax_action, eval_Z

            def target(self, observation: Union[np.ndarray, dict]):
                outputs = self.target_representation(observation)
                target_Z = self.target_Zhead(outputs['state'])
                target_Q = (self.supports * target_Z).sum(-1)
                argmax_action = target_Q.argmax(dim=-1)
                return outputs, argmax_action, target_Z

            def copy_target(self):
                for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.eval_Zhead.trainable_params(), self.target_Zhead.trainable_params()):
                    tp.assign_value(ep)


        class QRDQN_Network(nn.Cell):
            def __init__(self,
                         action_space: Discrete,
                         quantile_num: int,
                         representation: ModuleType,
                         hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(QRDQN_Network, self).__init__()
                self.action_dim = action_space.n
                self.quantile_num = quantile_num
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Zhead = QRDQNhead(self.representation.output_shapes['state'][0], self.action_dim, self.quantile_num,
                                            hidden_size,
                                            normalize, initialize, activation)
                self.target_Zhead = copy.deepcopy(self.eval_Zhead)

                self._mean = ms.ops.ReduceMean()

            def construct(self, observation: ms.tensor):
                outputs = self.representation(observation)
                evalZ = self.eval_Zhead(outputs['state'])
                evalQ = self._mean(evalZ, -1)
                argmax_action = evalQ.argmax(axis=-1)
                return outputs, argmax_action, evalZ

            def target(self, observation: ms.tensor):
                outputs = self.target_representation(observation)
                target_Z = self.target_Zhead(outputs['state'])
                target_Q = self._mean(target_Z, -1)
                argmax_action = target_Q.argmax(axis=-1)
                return outputs, argmax_action, target_Z

            def trainable_params(self, recurse=True):
                return self.representation.trainable_params() + self.eval_Zhead.trainable_params()

            def copy_target(self):
                for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.eval_Zhead.trainable_params(), self.target_Zhead.trainable_params()):
                    tp.assign_value(ep)


        class ActorNet(nn.Cell):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(ActorNet, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, nn.Tanh, initialize)[0])
                self.model = nn.SequentialCell(*layers)

            def construct(self, x: ms.tensor):
                return self.model(x)


        class CriticNet(nn.Cell):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(CriticNet, self).__init__()
                layers = []
                input_shape = (state_dim + action_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initialize)[0])
                self._concat = ms.ops.Concat(axis=-1)
                self.model = nn.SequentialCell(*layers)

            def construct(self, x: ms.tensor, a: ms.tensor):
                return self.model(self._concat((x, a)))[:, 0]


        class DDPGPolicy(nn.Cell):
            def __init__(self,
                         action_space: Space,
                         representation: ModuleType,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                assert isinstance(action_space, Box)
                super(DDPGPolicy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes

                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size, initialize,
                                      activation)
                self.critic = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                        initialize, activation)
                self.target_actor = copy.deepcopy(self.actor)
                self.target_critic = copy.deepcopy(self.critic)
                # options
                self._standard_normal = ms.ops.StandardNormal()
                self._min_act, self._max_act = ms.Tensor(-1.0), ms.Tensor(1.0)

            def construct(self, observation: ms.tensor):
                outputs = self.representation(observation)
                act = self.actor(outputs['state'])
                return outputs, act

            def Qtarget(self, observation: ms.tensor):
                outputs = self.representation(observation)
                act = self.target_actor(outputs['state'])
                return self.target_critic(outputs['state'], act)

            def Qaction(self, observation: ms.tensor, action: ms.tensor):
                outputs = self.representation(observation)
                return self.critic(outputs['state'], action)

            def Qpolicy(self, observation: ms.tensor):
                outputs = self.representation(observation)
                return self.critic(outputs['state'], self.actor(outputs['state']))

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor.trainable_params(), self.target_actor.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
                for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


        class TD3Policy(nn.Cell):
            def __init__(self,
                         action_space: Space,
                         representation: ModuleType,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(TD3Policy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                try:
                    self.representation_params = self.representation.trainable_params()
                except:
                    self.representation_params = []
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                      initialize, activation)
                self.criticA = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                         initialize, activation)
                self.criticB = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                         initialize, activation)
                self.target_actor = copy.deepcopy(self.actor)
                self.target_criticA = copy.deepcopy(self.criticA)
                self.target_criticB = copy.deepcopy(self.criticB)
                self.actor_params = self.representation_params + self.actor.trainable_params()
                # options
                self._standard_normal = ms.ops.StandardNormal()
                self._min_act, self._max_act = ms.Tensor(-1.0), ms.Tensor(1.0)
                self._minimum = ms.ops.Minimum()
                self._concat = ms.ops.Concat(axis=-1)
                self._expand_dims = ms.ops.ExpandDims()

            def action(self, observation: ms.tensor):
                outputs = self.representation(observation)
                act = self.actor(outputs['state'])
                return outputs, act

            def Qtarget(self, observation: ms.tensor):
                outputs = self.representation(observation)
                act = self.target_actor(outputs['state'])
                noise = ms.ops.clip_by_value(self._standard_normal(act.shape), self._min_act, self._max_act) * 0.1
                act = ms.ops.clip_by_value(act + noise, self._min_act, self._max_act)
                qa = self._expand_dims(self.target_criticA(outputs['state'], act), 1)
                qb = self._expand_dims(self.target_criticB(outputs['state'], act), 1)
                mim_q = self._minimum(qa, qb)
                return outputs, mim_q

            def Qaction(self, observation: ms.tensor, action: ms.tensor):
                outputs = self.representation(observation)
                qa = self._expand_dims(self.criticA(outputs['state'], action), 1)
                qb = self._expand_dims(self.criticB(outputs['state'], action), 1)
                return outputs, self._concat((qa, qb))

            def Qpolicy(self, observation: ms.tensor):
                outputs = self.representation(observation)
                act = self.actor(outputs['state'])
                qa = self._expand_dims(self.criticA(outputs['state'], act), 1)
                qb = self._expand_dims(self.criticB(outputs['state'], act), 1)
                return outputs, (qa + qb) / 2.0

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor.trainable_params(), self.target_actor.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
                for ep, tp in zip(self.criticA.trainable_params(), self.target_criticA.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
                for ep, tp in zip(self.criticB.trainable_params(), self.target_criticB.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


        class PDQNPolicy(nn.Cell):
            def __init__(self,
                         observation_space,
                         action_space,
                         representation: ModuleType,
                         conactor_hidden_size: Sequence[int],
                         qnetwork_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                super(PDQNPolicy, self).__init__()
                self.representation = representation
                self.observation_space = observation_space
                self.action_space = action_space
                self.num_disact = self.action_space.spaces[0].n
                self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
                self.conact_size = int(self.conact_sizes.sum())

                self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                           qnetwork_hidden_size, normalize,
                                           initialize, nn.ReLU)
                self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                         initialize, nn.ReLU)
                self.target_conactor = copy.deepcopy(self.conactor)
                self.target_qnetwork = copy.deepcopy(self.qnetwork)
                self._concat = ms.ops.Concat(1)

            def Atarget(self, state):
                target_conact = self.target_conactor(state)
                return target_conact

            def con_action(self, state):
                state = state.expand_dims(0).astype(ms.float32)
                conaction = self.conactor(state).squeeze()
                return conaction

            def Qtarget(self, state, action):
                input_q = self._concat((state, action))
                target_q = self.target_qnetwork(input_q)
                return target_q

            def Qeval(self, state, action):
                state = state.astype(ms.float32)
                input_q = self._concat((state, action))
                eval_q = self.qnetwork(input_q)
                return eval_q

            def Qpolicy(self, state):
                conact = self.conactor(state)
                input_q = self._concat((state, conact))
                policy_q = (self.qnetwork(input_q)).sum()
                return policy_q

            def construct(self):
                return super().construct()

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.conactor.trainable_params(), self.target_conactor.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
                for ep, tp in zip(self.qnetwork.trainable_params(), self.target_qnetwork.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


        class MPDQNPolicy(nn.Cell):
            def __init__(self,
                         observation_space,
                         action_space,
                         representation: ModuleType,
                         conactor_hidden_size: Sequence[int],
                         qnetwork_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                super(MPDQNPolicy, self).__init__()
                self.representation = representation
                self.observation_space = observation_space
                self.obs_size = self.observation_space.shape[0]
                self.action_space = action_space
                self.num_disact = self.action_space.spaces[0].n
                self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
                self.conact_size = int(self.conact_sizes.sum())

                self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                           qnetwork_hidden_size, normalize,
                                           initialize, nn.ReLU)
                self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                         initialize, nn.ReLU)
                self.target_conactor = copy.deepcopy(self.conactor)
                self.target_qnetwork = copy.deepcopy(self.qnetwork)

                self.offsets = self.conact_sizes.cumsum()
                self.offsets = np.insert(self.offsets, 0, 0)
                self.offsets = ms.Tensor(self.offsets)

                self._concat = ms.ops.Concat(1)
                self._zeroslike = ms.ops.ZerosLike()
                self._squeeze = ms.ops.Squeeze(1)

            def Atarget(self, state):
                target_conact = self.target_conactor(state)
                return target_conact

            def con_action(self, state):
                # conaction = self.conactor(state)
                state = state.expand_dims(0).astype(ms.float32)
                conaction = self.conactor(state).squeeze()
                return conaction

            def Qtarget(self, state, action):
                batch_size = state.shape[0]
                Q = []
                input_q = self._concat((state, self._zeroslike(action)))
                input_q = input_q.repeat(self.num_disact, 0)
                input_q = input_q.asnumpy()
                action = action.asnumpy()
                for i in range(self.num_disact):
                    input_q[i * batch_size:(i + 1) * batch_size,
                    self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                        = action[:, self.offsets[i]:self.offsets[i + 1]]
                input_q = ms.Tensor(input_q, dtype=ms.float32)
                eval_qall = self.target_qnetwork(input_q)
                for i in range(self.num_disact):
                    eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
                    if len(eval_q.shape) == 1:
                        eval_q = eval_q.expand_dims(1)
                    Q.append(eval_q)
                Q = self._concat(Q)
                return Q

            def Qeval(self, state, action, input_q):
                # state = state.astype(ms.float32)
                batch_size = state.shape[0]
                Q = []
                # input_q = self._concat((state, self._zeroslike(action)))
                # input_q = input_q.repeat(self.num_disact, 0)
                # input_q = input_q.asnumpy()
                # action = action.asnumpy()
                # for i in range(self.num_disact):
                #     input_q[i * batch_size:(i + 1) * batch_size, self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                #         = action[:, self.offsets[i]:self.offsets[i + 1]]
                #         # = self._squeeze(action[:, self.offsets[i]:self.offsets[i + 1]])
                # input_q = ms.Tensor(input_q, dtype=ms.float32)
                eval_qall = self.qnetwork(input_q)
                for i in range(self.num_disact):
                    eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
                    if len(eval_q.shape) == 1:
                        eval_q = eval_q.expand_dims(1)
                    Q.append(eval_q)
                Q = self._concat(Q)
                return Q

            def Qpolicy(self, state, input_q):
                # conact = self.conactor(state)
                batch_size = state.shape[0]
                Q = []
                # input_q = self._concat((state, self._zeroslike(conact)))
                # input_q = input_q.repeat(self.num_disact, 0)
                # for i in range(self.num_disact):
                #     input_q[i * batch_size:(i + 1) * batch_size,
                #     self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                #         = conact[:, self.offsets[i]:self.offsets[i + 1]]
                eval_qall = self.qnetwork(input_q)
                for i in range(self.num_disact):
                    eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
                    if len(eval_q.shape) == 1:
                        eval_q = eval_q.expand_dims(1)
                    Q.append(eval_q)
                Q = self._concat(Q)
                return Q

            def construct(self):
                return super().construct()

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.conactor.trainable_params(), self.target_conactor.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
                for ep, tp in zip(self.qnetwork.trainable_params(), self.target_qnetwork.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


        class SPDQNPolicy(nn.Cell):
            def __init__(self,
                         observation_space,
                         action_space,
                         representation: ModuleType,
                         conactor_hidden_size: Sequence[int],
                         qnetwork_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                super(SPDQNPolicy, self).__init__()
                self.representation = representation
                self.observation_space = observation_space
                self.obs_size = self.observation_space.shape[0]
                self.action_space = action_space
                self.num_disact = self.action_space.spaces[0].n
                self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
                self.conact_size = int(self.conact_sizes.sum())

                self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                           qnetwork_hidden_size, normalize,
                                           initialize, nn.ReLU)
                self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                         initialize, nn.ReLU)
                self.target_conactor = copy.deepcopy(self.conactor)
                self.target_qnetwork = copy.deepcopy(self.qnetwork)

                self.offsets = self.conact_sizes.cumsum()
                self.offsets = np.insert(self.offsets, 0, 0)
                self.offsets = ms.Tensor(self.offsets)

                self._concat = ms.ops.Concat(1)
                self._zeroslike = ms.ops.ZerosLike()
                self._squeeze = ms.ops.Squeeze(1)

            def Atarget(self, state):
                target_conact = self.target_conactor(state)
                return target_conact

            def con_action(self, state):
                state = state.expand_dims(0).astype(ms.float32)
                conaction = self.conactor(state).squeeze()
                return conaction

            def Qtarget(self, state, action):
                batch_size = state.shape[0]
                Q = []
                input_q = self._concat((state, self._zeroslike(action)))
                input_q = input_q.repeat(self.num_disact, 0)
                input_q = input_q.asnumpy()
                action = action.asnumpy()
                for i in range(self.num_disact):
                    input_q[i * batch_size:(i + 1) * batch_size,
                    self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                        = action[:, self.offsets[i]:self.offsets[i + 1]]
                input_q = ms.Tensor(input_q, dtype=ms.float32)
                eval_qall = self.target_qnetwork(input_q)
                for i in range(self.num_disact):
                    eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
                    if len(eval_q.shape) == 1:
                        eval_q = eval_q.expand_dims(1)
                    Q.append(eval_q)
                Q = self._concat(Q)
                return Q

            def Qeval(self, state, action, input_q):
                batch_size = state.shape[0]
                Q = []
                eval_qall = self.qnetwork(input_q)
                for i in range(self.num_disact):
                    eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
                    if len(eval_q.shape) == 1:
                        eval_q = eval_q.expand_dims(1)
                    Q.append(eval_q)
                Q = self._concat(Q)
                return Q

            def Qpolicy(self, state, input_q):
                batch_size = state.shape[0]
                Q = []
                eval_qall = self.qnetwork(input_q)
                for i in range(self.num_disact):
                    eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
                    if len(eval_q.shape) == 1:
                        eval_q = eval_q.expand_dims(1)
                    Q.append(eval_q)
                Q = self._concat(Q)
                return Q

            def construct(self):
                return super().construct()

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.conactor.trainable_params(), self.target_conactor.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
                for ep, tp in zip(self.qnetwork.trainable_params(), self.target_qnetwork.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


        class DRQNPolicy(nn.Cell):
            def __init__(self,
                         action_space: Discrete,
                         representation: nn.Cell,
                         **kwargs):
                super(DRQNPolicy, self).__init__()
                self.recurrent_layer_N = kwargs['recurrent_layer_N']
                self.rnn_hidden_dim = kwargs['recurrent_hidden_size']
                self.action_dim = action_space.n
                self.representation = representation
                self.target_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                kwargs["input_dim"] = self.representation.output_shapes['state'][0]
                kwargs["action_dim"] = self.action_dim
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.cnn = True if self.representation.cls_name == "Basic_CNN" else False
                self.eval_Qhead = BasicRecurrent(**kwargs)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)
                self._zeroslike = ms.ops.ZerosLike()

            def construct(self, observation: Union[np.ndarray, dict], *rnn_hidden: ms.tensor):
                if self.cnn:
                    obs_shape = observation.shape
                    outputs = self.representation(observation.reshape((-1,) + obs_shape[-3:]))
                    outputs['state'] = outputs['state'].reshape(obs_shape[0:-3] + (-1,))
                else:
                    outputs = self.representation(observation)
                if self.lstm:
                    hidden_states, cell_states, evalQ = self.eval_Qhead(outputs['state'], rnn_hidden[0], rnn_hidden[1])
                else:
                    hidden_states, evalQ = self.eval_Qhead(outputs['state'], rnn_hidden[0])
                    cell_states = None
                argmax_action = evalQ[:, -1].argmax(axis=-1)
                return outputs, argmax_action, evalQ, (hidden_states, cell_states)

            def target(self, observation: Union[np.ndarray, dict], *rnn_hidden: ms.tensor):
                if self.cnn:
                    obs_shape = observation.shape
                    outputs = self.representation(observation.reshape((-1,) + obs_shape[-3:]))
                    outputs['state'] = outputs['state'].reshape(obs_shape[0:-3] + (-1,))
                else:
                    outputs = self.representation(observation)
                if self.lstm:
                    hidden_states, cell_states, targetQ = self.target_Qhead(outputs['state'], rnn_hidden[0], rnn_hidden[1])
                else:
                    hidden_states, targetQ = self.target_Qhead(outputs['state'], rnn_hidden[0])
                    cell_states = None
                argmax_action = targetQ.argmax(axis=-1)
                return outputs, argmax_action, targetQ, (hidden_states, cell_states)

            def init_hidden(self, batch):
                hidden_states = ms.ops.zeros(size=(self.recurrent_layer_N, batch, self.rnn_hidden_dim))
                cell_states = self._zeroslike(hidden_states) if self.lstm else None
                return hidden_states, cell_states

            def init_hidden_item(self, rnn_hidden, i):
                if self.lstm:
                    rnn_hidden[0][:, i] = ms.ops.zeros(size=(self.recurrent_layer_N, self.rnn_hidden_dim))
                    rnn_hidden[1][:, i] = ms.ops.zeros(size=(self.recurrent_layer_N, self.rnn_hidden_dim))
                    return rnn_hidden
                else:
                    rnn_hidden[:, i] = ms.ops.zeros(size=(self.recurrent_layer_N, self.rnn_hidden_dim))
                    return rnn_hidden

            def copy_target(self):
                for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
                    tp.assign_value(ep)
                for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
                    tp.assign_value(ep)

