Deterministic-MARL
===================================================

This module defines several classes for different types of actor-networks, Q-networks, 
and policies in a multi-agent reinforcement learning setting. 

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.policies.deterministic_marl.BasicQhead(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation, device)

  This class defines the Q-value head with a neural network.
  It uses a multi-layer perceptron with a specified architecture defined by hidden_sizes.

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
  :return: The Q values of the input x.
  :rtype: torch.Tensor


.. py:class::
  xuance.torch.policies.deterministic_marl.BasicQnetwork(action_space, n_agents, representation, hidden_size, normalize, initialize, activation, device)

  The basic Q-network that is used to calculate the Q-values of observations.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.BasicQnetwork.forward(observation, agent_ids, *rnn_hidden, avail_actions=None)

  Performs a forward pass of the Q-network. 
  It takes an observation, agent IDs, and possibly recurrent hidden states, and produces Q-values.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :param rnn_hidden: The last recurrent hidden states.
  :param avail_actions: The masked values for availabel actions of each agent.
  :type avail_actions: torch.Tensor
  :return: A tuple that includes the new recurrent hidden states, greedy actions, and the evaluated Q-values of the observations for multiple agents.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.deterministic_marl.BasicQnetwork.target_Q(observation, agent_ids, *rnn_hidden)

  Computes the target Q-values for the given observation, agent IDs, and recurrent hidden states.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :return: A tuple that includes the new recurrent hidden states, and the target Q-values of the observations for multiple agents.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.deterministic_marl.BasicQnetwork.copy_target()

  Synchronize the target networks.


.. py:class::
  xuance.torch.policies.deterministic_marl.MFQnetwork(action_space, n_agents, representation, hidden_sizes, normalize, initialize, activation, device)

  An implementation of MFQ (Multi-Fidelity Q-network) model, which appears to be an extension or variation of the basic Q-network.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.MFQnetwork.forward(observation, actions_mean, agent_ids)

  Performs a forward pass of the MFQ-network. 
  It takes an observation, actions mean, agent IDs, and produces Q-values as inputs,
  and returns the outputs of the representation, the greedy actions, and the evaluated Q-values.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param actions_mean: The mean values of actions.
  :type actions_mean: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: A tuple that includes the outputs of the representation, the greedy actions, and the evaluated Q-values.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.deterministic_marl.MFQnetwork.sample_actions(logits)

  Given logits (output of the Q-network), samples actions from a categorical distribution.

  :param logits: The logits for categorical distributions.
  :type logits: torch.Tensor
  :return: sampled actions.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.deterministic_marl.MFQnetwork.target_Q(observation, actions_mean, agent_ids)

  Computes the target Q-values for the given observation, actions mean, and agent IDs.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param actions_mean: The mean values of actions.
  :type actions_mean: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: The target Q-values.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.deterministic_marl.MFQnetwork.copy_target()

  Synchronize the target networks.


.. py:class::
  xuance.torch.policies.deterministic_marl.MixingQnetwork(action_space, n_agents, representation, mixer, hidden_size, normalize, initialize, activation, device)

  Part of a multi-agent reinforcement learning setup for QMIX, VDN, or WQMIX algorithms. 
  This class appears to be an extension or modification of the BasicQnetwork class defined above.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
  :param hidden_size: The sizes of the hidden layers.
  :type hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.forward(observation, agent_ids, *rnn_hidden, avail_actions=None)

  Processes the input observation using the representation module.
  Concatenates the state and agent IDs.
  Computes the Q-values using the evaluation Q-head.
  Optionally masks unavailable actions.
  Returns hidden states, greedy action, and Q-values.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: torch.Tensor
  :return: A tuple that includes the hidden states, greedy action, and Q-values.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.target_Q(observation, agent_ids, *rnn_hidden)

  Similar to the forward method but uses target networks.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :return: A tuple that includes the hidden states, and Q-values.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.Q_tot(q, states)

  Compute the total Q-values using the evaluation mixers.

  :param q: The independent Q-values of n agents.
  :type q: torch.Tensor
  :param states: The global states.
  :type states: torch.Tensor
  :return: The total Q-values for the multi-agent team.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.target_Q_tot(q, states)

  Compute the total Q-values using the target mixers.

  :param q: The independent Q-values of n agents.
  :type q: torch.Tensor
  :param states: The global states.
  :type states: torch.Tensor
  :return: The total Q-values for the multi-agent team.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.copy_target()

  Synchronize the target networks.


.. py:class::
  xuance.torch.policies.deterministic_marl.Weighted_MixingQnetwork(action_space, n_agents, representation, mixer, ff_mixer, hidden_size, normalize, initialize, activation, device)

  This class is an implementation of Weight QMIX algorithms.
  It is an extention of the MixingQnetwork by introducing a centralized Q-value computation using a feedforward mixer. 
  It provides the necessary methods to calculate the Q-values for centralized evaluation and target networks.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
  :param ff_mixer: The feed forward mixer network.
  :type ff_mixer: nn.Module
  :param hidden_size: The sizes of the hidden layers.
  :type hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.Weighted_MixingQnetwork.q_centralized(observation, agent_ids, *rnn_hidden)

  Compute the centralized Q-values with the evaluation networks.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :return: The centralized Q-values.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.deterministic_marl.Weighted_MixingQnetwork.target_q_centralized(observation, agent_ids, *rnn_hidden)

  Compute the centralized Q-values with the target networks.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :return: The target centralized Q-values.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.deterministic_marl.Weighted_MixingQnetwork.copy_target()

  Synchronize the target networks.


.. py:class::
  xuance.torch.policies.deterministic_marl.Qtran_MixingQnetwork(action_space, n_agents, representation, mixer, qtran_mixer, hidden_size, normalize, initialize, activation, device)

  This class is an implementation of QTRAN algorithms.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
  :param qtran_mixer: The QTRAN mixer.
  :type qtran_mixer: nn.Module
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.Qtran_MixingQnetwork.forward(observation, agent_ids, *rnn_hidden, avail_actions=None)

  Processes the input observation using the representation module.
  Concatenates the state and agent IDs.
  Computes the Q-values using the evaluation Q-head.
  Optionally masks unavailable actions.
  Returns hidden states, greedy action, and Q-values.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :param rnn_hidden: The recurrent hidden states.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: torch.Tensor
  :return: A tuple that includes the new recurrent hiddenstates, the representation outputs, greedy action, and Q-values.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.deterministic_marl.Qtran_MixingQnetwork.target_Q(observation, agent_ids)

  Calculate the target Q-values of the agents team.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: A tuple that includes the new recurrent hiddenstates, the representation outputs, and the target Q-values.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.deterministic_marl.Qtran_MixingQnetwork.copy_target()

  Synchronize the target networks.


.. py:class::
  xuance.torch.policies.deterministic_marl.DCG_policy(action_space, global_state_dim, representation, utility, payoffs, dcgraph, hidden_size_bias, normalize, initialize, activation, device)

  An implementation of the policies of deep coordination graph (DCG) algorithm.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param global_state_dim: The dimension of the global state.
  :type global_state_dim: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param utility: The utility module used to calculate the utility values.
  :type utility: nn.Module
  :param payoffs: The payoffs module used to calculate the payoffs between agents.
  :type payoffs: nn.Module
  :param hidden_size_bias: The sizes of the bias hidden layer.
  :type hidden_size_bias: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.DCG_policy.forward(observation, agent_ids, *rnn_hidden, avail_actions=None)

  Computes the forward pass given an observation, agent IDs, and optional recurrent hidden states.
  Uses the representation module to obtain outputs.
  Returns new recurrent hidden states, greedy actions, and evaluated Q values.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: torch.Tensor
  :return: A tuple that includes the new recurrent hidden states, greedy actions, and evaluated Q values.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.deterministic_marl.DCG_policy.copy_target()

  Synchronize the target networks.


.. py:class::
  xuance.torch.policies.deterministic_marl.ActorNet(state_dim, n_agents, action_space, hidden_sizes, normalize, initialize, activation, device)

  A class that defines an actor network for MARL aglorithms based on deterministic policy gradient.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param action_space: The action space of the environment.
  :type action_space: Space
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.ActorNet.forward(x)

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: The determinsitc outputs of the actions.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.deterministic_marl.CriticNet(state_dim, n_agents, action_dim, hidden_sizes, normalize, initialize, activation, device)

  A class that defines an critic network for MARL aglorithms based on deterministic policy gradient.
  It is responsible for calculating the critic values of the states. 

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
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.CriticNet.forward()

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: The evaluated values of the inputs.
  :rtype: torch.Tensor


.. py:class::
  xuance.torch.policies.deterministic_marl.Basic_DDPG_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  An implementation of the basic policy for deep deterministic policy gradient (DDPG) algorithm.

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.Basic_DDPG_policy.forward(observation, agent_ids)

  A feed forward method that returns the representation outputs and deterministic actions given observations and agent IDs.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: A tuple that includes the outputs of the representation, and the deterministic actions.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.deterministic_marl.Basic_DDPG_policy.critic(observation, actions, agent_ids)

  A method that is used to calculate the Q values given observations, actions, and ID variables of agents.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param actions: The actions input.
  :type actions: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: The evaluated Q-values.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.deterministic_marl.Basic_DDPG_policy.target_critic(observation, actions, agent_ids)

  Similar to the method of self.critic() but with target critic networks.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param actions: The actions input.
  :type actions: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: The target Q-values.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.deterministic_marl.Basic_DDPG_policy.soft_update(tau=0.005)

  Performs a soft update of the target networks using a specified interpolation parameter (tau).

  :param tau: The soft update factor for the update of target networks, default is 0.005.
  :type tau: float


.. py:class::
  xuance.torch.policies.deterministic_marl.MADDPG_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  A class that is inherient from Basic_DDPG_policy.
  It is an implementation of the policy for multi-agent deep deterministic policy gradient (MADDPG) algorithm.

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.MADDPG_policy.critic(observation, actions, agent_ids)

  A method that is used to calculate the Q values given observations, actions, and ID variables of agents.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param actions: The actions input.
  :type actions: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: The Q-values that is calculated by evaluation critic networks.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.deterministic_marl.MADDPG_policy.target_critic(observation, actions, agent_ids)

  Similar to the self.critic method, but the values are calculated by target critic networks.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param actions: The actions input.
  :type actions: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: The target Q-values evaulated by target critic networks.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.deterministic_marl.MATD3_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  A class that is inherient from Basic_DDPG_policy.
  It is an implementation of the policy for multi-agent twine delayed deep deterministic policy gradient (MATD3) algorithm.

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.deterministic_marl.MATD3_policy.Qpolicy(observation, actions, agent_ids)

  A method that is used to calculate the Q-values by two critic networks.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param actions: The actions input.
  :type actions: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: A tuple that includes the representation outputs, and the evaulated Q-values by two critic networks.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.deterministic_marl.MATD3_policy.Qtarget(observation, actions, agent_ids)

  Similar to the self.Qpolicy() method, but the Q-values are calculated by target critic networks.
  Finally, it returns the minimum of two Q-values calculated by the two target critic networks, respectively.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param actions: The actions input.
  :type actions: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: A tuple that includes the representation outputs, and the minimum of target Q-values calculted by two target critic networks.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.deterministic_marl.MATD3_policy.Qaction(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param actions: The actions input.
  :type actions: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: A tuple that includes the representation outputs, and the concatenates of evaluated Q-values calculted by two critic networks.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.deterministic_marl.MATD3_policy.soft_update(tau=0.005)

  Performs a soft update of the target networks using a specified interpolation parameter (tau).

  :param tau: The soft update factor for the update of target networks, default is 0.005.
  :type tau: float

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.BasicQhead(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation, device)

  This class defines the Q-value head with a neural network. 
  It uses a multi-layer perceptron with a specified architecture defined by hidden_sizes.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.BasicQhead.call(x)

  :param x: The input tensor.
  :type x: tf.Tensor
  :return: The Q values of the input x.
  :rtype: tf.Tensor


.. py:class::
  xuance.tensorflow.policies.deterministic_marl.BasicQnetwork(action_space, n_agents, representation, hidden_size, normalize, initialize, activation, device)

  The basic Q-network that is used to calculate the Q-values of observations.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: tk.Model
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.BasicQnetwork.call(inputs, *rnn_hidden, avail_actions=None)

  Performs a forward pass of the Q-network. 
  It takes an observation, agent IDs, and possibly recurrent hidden states, and produces Q-values.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :param rnn_hidden: The final hidden state of the sequence.
  :param avail_actions: The masked values for availabel actions of each agent.
  :type avail_actions: tf.Tensor
  :return: A tuple that includes the new recurrent hidden states, greedy actions, and the evaluated Q-values of the observations for multiple agents.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.BasicQnetwork.target_Q(inputs, *rnn_hidden)

  Computes the target Q-values for the given observation, agent IDs, and recurrent hidden states.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :param rnn_hidden: The final hidden state of the sequence.
  :return: A tuple that includes the new recurrent hidden states, and the target Q-values of the observations for multiple agents.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.BasicQnetwork.trainable_param()

  :return: trainable parameters of the networks.

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.BasicQnetwork.copy_target()

  Synchronize the target networks.


.. py:class::
  xuance.tensorflow.policies.deterministic_marl.MFQnetwork(action_space, n_agents, representation, hidden_sizes, normalize, initialize, activation, device)

  An implementation of MFQ (Multi-Fidelity Q-network) model, which appears to be an extension or variation of the basic Q-network.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: tk.Model
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MFQnetwork.call(inputs)

  Performs a forward pass of the MFQ-network. 
  It takes an observation, actions mean, agent IDs, and produces Q-values as inputs, and returns the outputs of the representation, 
  the greedy actions, and the evaluated Q-values.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: A tuple that includes the outputs of the representation, the greedy actions, and the evaluated Q-values.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MFQnetwork.sample_actions(logits)

  Given logits (output of the Q-network), samples actions from a categorical distribution.

  :param logits: The logits for categorical distributions.
  :type logits: tf.Tensor
  :return: sampled actions.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MFQnetwork.target_Q(observation, actions_mean, agent_ids)

  Computes the target Q-values for the given observation, actions mean, and agent IDs.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param actions_mean: The mean values of actions.
  :type actions_mean: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :return: The target Q-values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MFQnetwork.copy_target()

  Synchronize the target networks.


.. py:class::
  xuance.tensorflow.policies.deterministic_marl.MixingQnetwork(action_space, n_agents, representation, mixer, hidden_size, normalize, initialize, activation, device)

  Part of a multi-agent reinforcement learning setup for QMIX, VDN, or WQMIX algorithms. 
  This class appears to be an extension or modification of the BasicQnetwork class defined above.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: tk.Model
  :param mixer: The mixer for independent values.
  :type mixer: tk.Model
  :param hidden_size: The sizes of the hidden layers.
  :type hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MixingQnetwork.call(inputs, *rnn_hidden)

  Processes the input observation using the representation module. 
  Concatenates the state and agent IDs. Computes the Q-values using the evaluation Q-head. 
  Optionally masks unavailable actions. Returns hidden states, greedy action, and Q-values

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :return: A tuple that includes the hidden states, greedy action, and Q-values.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MixingQnetwork.target_Q(inputs)

  Similar to the forward method but uses target networks.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: A tuple that includes the hidden states, and Q-values.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MixingQnetwork.Q_tot(q, states)

  Compute the total Q-values using the evaluation mixers.

  :param q: The independent Q-values of n agents.
  :type q: tf.Tensor
  :param states: The global states.
  :type states: tf.Tensor
  :return: The total Q-values for the multi-agent team.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MixingQnetwork.target_Q_tot(q, states)

  Compute the total Q-values using the target mixers.

  :param q: The independent Q-values of n agents.
  :type q: tf.Tensor
  :param states: The global states.
  :type states: tf.Tensor
  :return: The total Q-values for the multi-agent team.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MixingQnetwork.copy_target()

  Synchronize the target networks.


.. py:class::
  xuance.tensorflow.policies.deterministic_marl.Weighted_MixingQnetwork(action_space, n_agents, representation, mixer, ff_mixer, hidden_size, normalize, initialize, activation, device)

  This class is an implementation of Weight QMIX algorithms. 
  It is an extention of the MixingQnetwork by introducing a centralized Q-value computation using a feedforward mixer. 
  It provides the necessary methods to calculate the Q-values for centralized evaluation and target networks

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: tk.Model
  :param mixer: The mixer for independent values.
  :type mixer: tk.Model
  :param ff_mixer: The feed forward mixer network.
  :type ff_mixer: tk.Model
  :param hidden_size: The sizes of the hidden layers.
  :type hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Weighted_MixingQnetwork.q_centralized(inputs, *rnn_hidden)

  Compute the centralized Q-values with the evaluation networks.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :param rnn_hidden: The last final hidden states of the sequence.
  :return: The centralized Q-values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Weighted_MixingQnetwork.target_q_centralized(inputs, *rnn_hidden)

  Compute the centralized Q-values with the target networks.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :param rnn_hidden: The last final hidden states of the sequence.
  :return: The target centralized Q-values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Weighted_MixingQnetwork.copy_target()

  Synchronize the target networks.


.. py:class::
  xuance.tensorflow.policies.deterministic_marl.Qtran_MixingQnetwork(action_space, n_agents, representation, mixer, qtran_mixer, hidden_size, normalize, initialize, activation, device)

  This class is an implementation of QTRAN algorithms.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: tk.Model
  :param mixer: The mixer for independent values.
  :type mixer: tk.Model
  :param qtran_mixer: The QTRAN mixer.
  :type qtran_mixer: tk.Model
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Qtran_MixingQnetwork.call(inputs, *rnn_hidden, avail_actions=None)

  Processes the input observation using the representation module. 
  Concatenates the state and agent IDs. Computes the Q-values using the evaluation Q-head. 
  Optionally masks unavailable actions. Returns hidden states, greedy action, and Q-values.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :param rnn_hidden: The recurrent hidden states.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: tf.Tensor
  :return: A tuple that includes the new recurrent hiddenstates, the representation outputs, greedy action, and Q-values.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Qtran_MixingQnetwork.target_Q(inputs)

  Calculate the target Q-values of the agents team.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: A tuple that includes the new recurrent hiddenstates, the representation outputs, and the target Q-values.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Qtran_MixingQnetwork.copy_target()

  Synchronize the target networks.


.. py:class::
  xuance.tensorflow.policies.deterministic_marl.DCG_policy(action_space, global_state_dim, representation, utility, payoffs, dcgraph, hidden_size_bias, normalize, initialize, activation, device)

  An implementation of the policies of deep coordination graph (DCG) algorithm.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param global_state_dim: The dimension of the global state.
  :type global_state_dim: int
  :param representation: The representation module.
  :type representation: tk.Model
  :param utility: The utility module used to calculate the utility values.
  :type utility: tk.Model
  :param payoffs: The payoffs module used to calculate the payoffs between agents.
  :type payoffs: tk.Model
  :param hidden_size_bias: The sizes of the bias hidden layer.
  :type hidden_size_bias: list
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.DCG_policy.call(inputs, *rnn_hidden, avail_actions=None)

  Computes the forward pass given an observation, agent IDs, and optional recurrent hidden states. 
  Uses the representation module to obtain outputs. 
  Returns new recurrent hidden states, greedy actions, and evaluated Q values.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :param rnn_hidden: The last final hidden states of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: tf.Tensor
  :return: A tuple that includes the new recurrent hidden states, greedy actions, and evaluated Q values.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.DCG_policy.copy_target()

  Synchronize the target networks.
  

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.ActorNet(state_dim, n_agents, action_space, hidden_sizes, normalize, initialize, activation, device)

  A class that defines an actor network for MARL aglorithms based on deterministic policy gradient.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param action_space: The action space of the environment.
  :type action_space: Space
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.ActorNet.call(x)

  :param x: The input tensor.
  :type x: tf.Tensor
  :return: The determinsitc outputs of the actions.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.CriticNet(independent, state_dim, n_agents, action_dim, hidden_sizes, normalize, initialize, activation, device)

  A class that defines an critic network for MARL aglorithms based on deterministic policy gradient. 
  It is responsible for calculating the critic values of the states.

  :param independent: Determine whether to calculate independent values.
  :type independent: bool
  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.CriticNet.call(x)

  The evaluated values of the inputs.

  :param x: The input tensor.
  :type x: tf.Tensor
  :return: The evaluated values of the inputs.
  :rtype: tf.Tensor


.. py:class::
  xuance.tensorflow.policies.deterministic_marl.Basic_DDPG_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  An implementation of the basic policy for deep deterministic policy gradient (DDPG) algorithm.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: tk.Model
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Basic_DDPG_policy.call(inputs)

  A feed forward method that returns the representation outputs and deterministic actions given observations and agent IDs.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: A tuple that includes the outputs of the representation, and the deterministic actions.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Basic_DDPG_policy.critic(observation, actions, agent_ids)

  A method that is used to calculate the Q values given observations, actions, and ID variables of agents.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param actions: The actions input.
  :type actions: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :return: The evaluated Q-values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Basic_DDPG_policy.target_critic(observation, actions, agent_ids)

  Similar to the method of self.critic() but with target critic networks.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param actions: The actions input.
  :type actions: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :return: The target Q-values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.Basic_DDPG_policy.soft_update(tau)

  Performs a soft update of the target networks using a specified interpolation parameter (tau).

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: The soft update factor for the update of target networks, default is 0.005.
  :rtype: float

.. py:class::
  xuance.tensorflow.policies.deterministic_marl.MADDPG_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  A class that is inherient from Basic_DDPG_policy. 
  It is an implementation of the policy for multi-agent deep deterministic policy gradient (MADDPG) algorithm.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: tk.Model
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MADDPG_policy.critic(observation, actions, agent_ids)

  A method that is used to calculate the Q values given observations, actions, and ID variables of agents.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param actions: The actions input.
  :type actions: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :return: The Q-values that is calculated by evaluation critic networks.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MADDPG_policy.target_critic(observation, actions, agent_ids)

  Similar to the self.critic method, but the values are calculated by target critic networks.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param actions: The actions input.
  :type actions: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :return: The target Q-values evaulated by target critic networks.
  :rtype: tf.Tensor


.. py:class::
  xuance.tensorflow.policies.deterministic_marl.MATD3_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  A class that is inherient from Basic_DDPG_policy. 
  It is an implementation of the policy for multi-agent twine delayed deep deterministic policy gradient (MATD3) algorithm.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: tk.Model
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MATD3_policy.call(inputs)

  A feed forward method that returns the representation outputs and deterministic actions given observations and agent IDs.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: A tuple that includes the outputs of the representation, and the deterministic actions.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MATD3_policy.critic(observation, actions, agent_ids)

  A method that is used to calculate the Q values given observations, actions, and ID variables of agents.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param actions: The actions input.
  :type actions: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :return: The evaluated Q-values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MATD3_policy.target_critic(observation, actions, agent_ids)

  Similar to the method of self.critic() but with target critic networks.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param actions: The actions input.
  :type actions: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :return: The target Q-values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MATD3_policy.Qaction(observation, actions, agent_ids)

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param actions: The actions input.
  :type actions: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :return: A tuple that includes the representation outputs, and the concatenates of evaluated Q-values calculted by two critic networks.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MATD3_policy.target_actor(inputs)

  A method used to calculate the actions with target actor networks.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: The target actions.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.deterministic_marl.MATD3_policy.soft_update(tau)

  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.policies.deterministic_marl.BasicQhead(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation)

  This class defines the Q-value head with a neural network. 
  It uses a multi-layer perceptron with a specified architecture defined by hidden_sizes.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell

.. py:function::
  xuance.mindspore.policies.deterministic_marl.BasicQhead.construct(x)

  :param x: The input tensor.
  :type x: ms.Tensor
  :return: The Q values of the input x.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.deterministic_marl.BasicQnetwork(action_space, n_agents, representation, hidden_size, normalize, initialize, activation, kwargs)

  The basic Q-network that is used to calculate the Q-values of observations.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Cell
  :param hidden_size: The sizes of the hidden layers.
  :type hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell
  :param kwargs: The other arguments.
  :type kwargs: dict

.. py:function::
  xuance.mindspore.policies.deterministic_marl.BasicQnetwork.construct(observation, agent_ids, *rnn_hidden, avail_actions=None)

  Performs a forward pass of the Q-network. 
  It takes an observation, agent IDs, and possibly recurrent hidden states, and produces Q-values.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: ms.Tensor
  :return: A tuple that includes the new recurrent hidden states, greedy actions, and the evaluated Q-values of the observations for multiple agents.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.deterministic_marl.BasicQnetwork.target_Q(observation, agent_ids, *rnn_hidden)

  Computes the target Q-values for the given observation, agent IDs, and recurrent hidden states.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :return: A tuple that includes the new recurrent hidden states, and the target Q-values of the observations for multiple agents.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.deterministic_marl.BasicQnetwork.trainable_params(recurse)

  Get trainable parameters of the models.

.. py:function::
  xuance.mindspore.policies.deterministic_marl.BasicQnetwork.copy_target()

  Synchronize the target networks..


.. py:class::
  xuance.mindspore.policies.deterministic_marl.MFQnetwork(action_space, n_agents, representation, hidden_size, normalize, initialize, activation)

  An implementation of MFQ (Multi-Fidelity Q-network) model, which appears to be an extension or variation of the basic Q-network.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Cell
  :param hidden_size: The sizes of the hidden layers.
  :type hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MFQnetwork.construct(observation, actions_mean, agent_ids)

  Performs a forward pass of the MFQ-network. 
  It takes an observation, actions mean, agent IDs, and produces Q-values as inputs, 
  and returns the outputs of the representation, the greedy actions, and the evaluated Q-values.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param actions_mean: The mean values of actions.
  :type actions_mean: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: A tuple that includes the outputs of the representation, the greedy actions, and the evaluated Q-values.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MFQnetwork.sample_actions(logits)

  Given logits (output of the Q-network), samples actions from a categorical distribution.

  :param logits: The logits for categorical distributions.
  :type logits: ms.Tensor
  :return: Sampled actions.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MFQnetwork.target_Q(observation, actions_mean, agent_ids)

  Computes the target Q-values for the given observation, actions mean, and agent IDs.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param actions_mean: The mean values of actions.
  :type actions_mean: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The target Q-values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MFQnetwork.copy_target()

  Synchronize the target networks.


.. py:class::
  xuance.mindspore.policies.deterministic_marl.MixingQnetwork(action_space, n_agents, representation, mixer, hidden_size, normalize, initialize, activation, kwargs)

  Part of a multi-agent reinforcement learning setup for QMIX, VDN, or WQMIX algorithms. 
  This class appears to be an extension or modification of the BasicQnetwork class defined above

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Cell
  :param mixer: The mixer for independent values.
  :type mixer: nn.Cell
  :param hidden_size: The sizes of the hidden layers.
  :type hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell
  :param kwargs: The other arguments.
  :type kwargs: dict

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MixingQnetwork.construct(observation, agent_ids, *rnn_hidden, avail_actions=None)

  Processes the input observation using the representation module. 
  Concatenates the state and agent IDs. 
  Computes the Q-values using the evaluation Q-head. 
  Optionally masks unavailable actions. 
  Returns hidden states, greedy action, and Q-values.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: ms.Tensor
  :return: A tuple that includes the hidden states, greedy action, and Q-values.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MixingQnetwork.target_Q(observation, agent_ids, *rnn_hidden)

  Similar to the forward method but uses target networks.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :return: A tuple that includes the hidden states, and Q-values.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MixingQnetwork.Q_tot(q, state)

  Compute the total Q-values using the evaluation mixers.

  :param q: The independent Q-values of n agents.
  :type q: ms.Tensor
  :param state: The global states.
  :type state: ms.Tensor
  :return: The total Q-values for the multi-agent team.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MixingQnetwork.target_Q_tot(q, state)

  Compute the total Q-values using the target mixers.

  :param q: The independent Q-values of n agents.
  :type q: ms.Tensor
  :param state: The global states.
  :type state: ms.Tensor
  :return: The total Q-values for the multi-agent team.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MixingQnetwork.trainable_params(recurse)

  Get trainable parameters of the models.

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MixingQnetwork.copy_target()

  Synchronize the target networks.


.. py:class::
  xuance.mindspore.policies.deterministic_marl.Weighted_MixingQnetwork(action_space, n_agents, representation, mixer, ff_mixer, hidden_size, normalize, initialize, activation, kwargs)

  This class is an implementation of Weight QMIX algorithms. 
  It is an extention of the MixingQnetwork by introducing a centralized Q-value computation using a feedforward mixer. 
  It provides the necessary methods to calculate the Q-values for centralized evaluation and target networks.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Cell
  :param mixer: The mixer for independent values.
  :type mixer: nn.Cell
  :param ff_mixer: The feed forward mixer network.
  :type ff_mixer: nn.Cell
  :param hidden_size: The sizes of the hidden layers.
  :type hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell
  :param kwargs: The other arguments.
  :type kwargs: dict

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Weighted_MixingQnetwork.q_centralized(observation, agent_ids, *rnn_hidden)

  Compute the centralized Q-values with the evaluation networks.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :return: The centralized Q-values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Weighted_MixingQnetwork.target_q_centralized(observation, agent_ids, *rnn_hidden)

  Compute the centralized Q-values with the target networks.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :return: The target centralized Q-values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Weighted_MixingQnetwork.copy_target()

  Synchronize the target networks..


.. py:class::
  xuance.mindspore.policies.deterministic_marl.Qtran_MixingQnetwork(action_space, n_agents, representation, mixer, qtran_mixer, hidden_size, normalize, initialize, activation, kwargs)

  This class is an implementation of QTRAN algorithms.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Cell
  :param mixer: The mixer for independent values.
  :type mixer: nn.Cell
  :param qtran_mixer: The QTRAN mixer.
  :type qtran_mixer: nn.Cell
  :param hidden_size: The sizes of the hidden layers.
  :type hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell
  :param kwargs: The other arguments.
  :type kwargs: dict

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Qtran_MixingQnetwork.construct(observation, agent_ids, *rnn_hidden, avail_actions=None)

  Processes the input observation using the representation module. 
  Concatenates the state and agent IDs. 
  Computes the Q-values using the evaluation Q-head. 
  Optionally masks unavailable actions. 
  Returns hidden states, greedy action, and Q-values.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: ms.Tensor
  :return: A tuple that includes the new recurrent hiddenstates, the representation outputs, greedy action, and Q-values.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Qtran_MixingQnetwork.target_Q(observation, agent_ids, *rnn_hidden)

  Calculate the target Q-values of the agents team.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :return: A tuple that includes the new recurrent hiddenstates, the representation outputs, and the target Q-values.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Qtran_MixingQnetwork.copy_target()

  Synchronize the target networks.
  

.. py:class::
  xuance.mindspore.policies.deterministic_marl.DCG_policy(action_space, global_state_dim, representation, utility, payoffs, dcgraph, hidden_size_bias, normalize, initialize, activation, kwargs)

  An implementation of the policies of deep coordination graph (DCG) algorithm.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param global_state_dim: The dimension of the global state.
  :type global_state_dim: int
  :param representation: The representation module.
  :type representation: nn.Cell
  :param utility: The utility module used to calculate the utility values.
  :type utility: nn.Cell
  :param payoffs: The payoffs module used to calculate the payoffs between agents.
  :type payoffs: nn.Cell
  :param dcgraph: The deep coordinatino graph.
  :type dcgraph: nn.Module
  :param hidden_size_bias: The sizes of the bias hidden layer.
  :type hidden_size_bias: int
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell
  :param kwargs: The other arguments.
  :type kwargs: dict

.. py:function::
  xuance.mindspore.policies.deterministic_marl.DCG_policy.construct(observation, agent_ids, *rnn_hidden, avail_actions=None)

  Computes the forward pass given an observation, agent IDs, and optional recurrent hidden states. 
  Uses the representation module to obtain outputs. 
  Returns new recurrent hidden states, greedy actions, and evaluated Q values.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: ms.Tensor
  :return: A tuple that includes the new recurrent hidden states, greedy actions, and evaluated Q values.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.deterministic_marl.DCG_policy.copy_target()

  Synchronize the target networks..


.. py:class::
  xuance.mindspore.policies.deterministic_marl.ActorNet(state_dim, n_agents, action_dim, hidden_sizes, normalize, initialize, activation)

  A class that defines an actor network for MARL aglorithms based on deterministic policy gradient.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell

.. py:function::
  xuance.mindspore.policies.deterministic_marl.ActorNet.construct(x)

  :param x: The input tensor.
  :type x: ms.Tensor
  :return: The determinsitc outputs of the actions.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.deterministic_marl.CriticNet(independent, state_dim, n_agents, action_dim, hidden_sizes, normalize, initialize, activation)

  A class that defines an critic network for MARL aglorithms based on deterministic policy gradient. 
  It is responsible for calculating the critic values of the states.

  :param independent: Determine whether to calculate independent values.
  :type independent: bool
  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell

.. py:function::
  xuance.mindspore.policies.deterministic_marl.CriticNet.construct(x)

  The evaluated values of the inputs.

  :param x: The input tensor.
  :type x: ms.Tensor
  :return: The evaluated values of the inputs.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.deterministic_marl.Basic_DDPG_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  An implementation of the basic policy for deep deterministic policy gradient (DDPG) algorithm

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Cell
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Basic_DDPG_policy.construct(observation, agent_ids)

  A feed forward method that returns the representation outputs and deterministic actions given observations and agent IDs.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: A tuple that includes the outputs of the representation, and the deterministic actions.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Basic_DDPG_policy.critic(observation, action, agent_ids)

  A method that is used to calculate the Q values given observations, actions, and ID variables of agents.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param action: The actions input.
  :type action: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The evaluated Q-values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Basic_DDPG_policy.target_critic(observation, action, agent_ids)

  Similar to the method of self.critic() but with target critic networks.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param action: The actions input.
  :type action: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The target Q-values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Basic_DDPG_policy.target_actor(observation, agent_ids)

  A method used to calculate the actions with target actor networks.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The target actions.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.deterministic_marl.Basic_DDPG_policy.soft_update(tau)

  Performs a soft update of the target networks using a specified interpolation parameter (tau).

  :param tau: The soft update factor for the update of target networks.
  :type tau: float


.. py:class::
  xuance.mindspore.policies.deterministic_marl.MADDPG_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  A class that is inherient from Basic_DDPG_policy. 
  It is an implementation of the policy for multi-agent deep deterministic policy gradient (MADDPG) algorithm.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Cell
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MADDPG_policy.construct(observation, agent_ids)

  A feed forward method that returns the representation outputs and deterministic actions given observations and agent IDs.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: A tuple that includes the outputs of the representation, and the deterministic actions.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MADDPG_policy.critic(observation, action, agent_ids)

  A method that is used to calculate the Q values given observations, actions, and ID variables of agents.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param action: The actions input.
  :type action: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The evaluated Q-values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MADDPG_policy.target_critic(observation, action, agent_ids)

  Similar to the method of self.critic() but with target critic networks.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param action: The actions input.
  :type action: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The target Q-values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MADDPG_policy.target_actor(observation, agent_ids)

  A method used to calculate the actions with target actor networks.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The target actions.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MADDPG_policy.soft_update(tau)

  :param tau: The soft update factor for the update of target networks.
  :type tau: float


.. py:class::
  xuance.mindspore.policies.deterministic_marl.MATD3_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  A class that is inherient from Basic_DDPG_policy. 
  It is an implementation of the policy for multi-agent twine delayed deep deterministic policy gradient (MATD3) algorithm

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Cell
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param critic_hidden_size: The sizes of the hidden layers in critic networks.
  :type critic_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MATD3_policy.Qpolicy(observation, action, agent_ids)

  A method that is used to calculate the Q-values by two critic networks.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param action: The actions input.
  :type action: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: A tuple that includes the representation outputs, and the evaulated Q-values by two critic networks.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MATD3_policy.Qtarget(observation, action, agent_ids)

  Similar to the self.Qpolicy() method, but the Q-values are calculated by target critic networks. 
  Finally, it returns the minimum of two Q-values calculated by the two target critic networks, respectively.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param action: The actions input.
  :type action: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: A tuple that includes the representation outputs, and the minimum of target Q-values calculted by two target critic networks.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MATD3_policy.Qaction_A(observation, action, agent_ids)

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param action: The actions input.
  :type action: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: A tuple that includes the representation outputs, and the evaluated Q-values calculted by critic-A networks.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MATD3_policy.Qaction_B(observation, action, agent_ids)

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param action: The actions input.
  :type action: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: A tuple that includes the representation outputs, and the evaluated Q-values calculted by critic-B networks.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.deterministic_marl.MATD3_policy.soft_update(tau)

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
                        utility: Optional[tk.Model] = None,
                        payoffs: Optional[tk.Model] = None,
                        dcgraph: Optional[tk.Model] = None,
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

            def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden: tf.Tensor, **kwargs):
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
                          *rnn_hidden: ms.Tensor, avail_actions=None):
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

            def target_Q(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: ms.Tensor):
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
                          *rnn_hidden: ms.Tensor, avail_actions=None):
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

            def target_Q(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: ms.Tensor):
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

            def q_centralized(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: ms.Tensor):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                else:
                    outputs = self.representation(observation)
                q_inputs = self._concat([outputs['state'], agent_ids])
                return self.eval_Qhead_centralized(q_inputs)

            def target_q_centralized(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: ms.Tensor):
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
                          *rnn_hidden: ms.Tensor, avail_actions=None):
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

            def target_Q(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: ms.Tensor):
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
                        utility: Optional[nn.Cell] = None,
                        payoffs: Optional[nn.Cell] = None,
                        dcgraph: Optional[nn.Cell] = None,
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
                          *rnn_hidden: ms.Tensor, avail_actions=None):
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
                        initialize: Optional[Callable[..., ms.Tensor]] = None,
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

