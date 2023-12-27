Categorical-MARL
======================================

Here, we provide several neural network modules for categorical MARL policies with discrete action space.
Each policy can generate actions from a categorical distribution, or generate a categorical distribution directly.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.policies.categorical_mal.ActorNet(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, gain, activation, device)
  
  The actor network that is used to calculate the logits of categorical distributions.

  :param state_dim: The dimension of the state varibale.
  :type state_dim: int
  :param action_dim: The dimentin of the actions.
  :type action_dim: int
  :param n_agents: The number of all agents.
  :type n_agents: int
  :param hidden_sizes: The number of hidden units for hidden layers.
  :type hidden_sizes: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: torch.Tensor
  :param gain: optional scaling factor.
  :type gain: float
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The logits for categorical distributions.
  :type device: str

.. py:function::
  xuance.torch.policies.categorical_mal.ActorNet.forward(x)

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: The logits for the policy.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.categorical_mal.CriticNet(state_dim, n_agents, hidden_sizes,  normalize, initialize, activation, device)

  The critic network that is used to evaluate the states or state-action pairs.

  :param state_dim: The dimension of the state varibale.
  :type state_dim: int
  :param n_agents: number of agents.
  :type n_agents: int
  :param hidden_sizes: The number of hidden units for hidden layers.
  :type hidden_sizes: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.categorical_mal.CriticNet.forward(x)

  :param x: The input data.
  :type x: torch.Tensor
  :return: The evaluated critic value.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.categorical_mal.COMA_Critic(state_dim, act_dim, hidden_sizes,  normalize, initialize, activation, device)

  The critic network for the Counterfactual Multi-Agent (COMA) algorithm, 
  which is used to evaluate the global state of the multi-agent systems.

  :param state_dim: The dimension of the state varibale.
  :type state_dim: int
  :param act_dim: The dimension of actions.
  :type act_dim: int
  :param hidden_sizes: The number of hidden units for hidden layers.
  :type hidden_sizes: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.categorical_mal.COMA_Critic.forward(x)

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: The evaluated critic value.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.categorical_mal.MAAC_Policy(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  Multi-Agent Actor-Critic (MAAC) policy for discrete action space. 
  It is used to generate categorical distributions and the total values of the multi-agent team.

  :param action_space: The action space.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
  :param actor_hidden_size: The number of hidden units for actor's hidden layers.
  :type actor_hidden_size: list
  :param critic_hidden_size: The number of hidden units for critic's hidden layers.
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
  xuance.torch.policies.categorical_mal.MAAC_Policy.forward(observation, agent_ids, *rnn_hidden, avail_actions=None)

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: torch.Tensor
  :return: A tuple that includes the final rnn hidden state, and the stochastic policies.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.categorical_mal.MAAC_Policy.get_values(critic_in, agent_ids, *rnn_hidden)

  Get the critic values of the agents.

  :param critic_in: The input variables of critic networks.
  :type critic_in: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :return: A tuple that includes the final rnn hidden state, and the total values of the team.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.categorical_mal.MAAC_Policy.value_tot(values_n, global_state)

  Calculate the total values of the team via the value decomposition modules (self.mixer).

  :param values_n: The joint values of n agents.
  :type values_n: torch.Tensor
  :param global_state: The global states of the environments.
  :type global_state: torch.Tensor
  :return: The final rnn hidden state, and the total values of the team.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.categorical_mal.MAAC_Policy_Share(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  Similar to MAAC_Policy but shares representations between agents.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
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
  xuance.torch.policies.categorical_mal.MAAC_Policy_Share.forward(observation, agent_ids, *rnn_hidden, avail_actions=None)

  :param observation: The original observation variables.
  :type observation: Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: torch.Tensor
  :return: A tuple that includes the final rnn hidden state, the stochastic policies, and the total values of the team.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.categorical_mal.MAAC_Policy_Share.value_tot(values_n, global_state)

  Calculate the total values of the team via the value decomposition modules (self.mixer).

  :param values_n: The joint values of n agents.
  :type values_n: Tensor
  :param global_state: The global states of the environments.
  :type global_state: Tensor
  :return: A tuple that includes the final rnn hidden state, and the total values of the team.
  :rtype: tuple

.. py:class::
  xuance.torch.policies.categorical_mal.COMAPolicy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  A policy for the Counterfactual Multi-Agent Policy (COMA) algorithm.

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
  xuance.torch.policies.categorical_mal.COMAPolicy.forward(observation, agent_ids, *rnn_hidden, avail_actions=None)

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: torch.Tensor
  :return: A tuple that includes the final rnn hidden state, and the action probabilities.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.categorical_mal.COMAPolicy.get_values(critic_in, *rnn_hidden, target=False)

  :param critic_in: The input variables of critic networks.
  :type critic_in: torch.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :param target: Determine whether to use target networks to calculate the values.
  :type target: bool
  :return: The team values.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.categorical_mal.COMAPolicy.copy_target()

  Synchronize the target networks.

.. py:class::
  xuance.torch.policies.categorical_mal.MeanFieldActorCriticPolicy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device, **kwargs)
  
  Mean Field Actor-Critic policy.

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
  :param kwargs: The other arguments.
  :type kwargs: dict

.. py:function::
  xuance.torch.policies.categorical_mal.MeanFieldActorCriticPolicy.forward(observation, agent_ids)

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: A tuple of tensors that includes the outputs hidden states and the categorical distributions.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.categorical_mal.MeanFieldActorCriticPolicy.critic(observation, actions_mean, agent_ids)

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param actions_mean: The mean values of actions.
  :type actions_mean: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: The evaluated values of the observations and mean-actions pairs.
  :rtype: torch.Tensor

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.policies.categorical_mal.ActorNet(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, gain, activation, device)
  
  The actor network that is used to calculate the logits of categorical distributions

  :param state_dim: The dimension of the state varibale.
  :type state_dim: int
  :param action_dim: The dimentin of the actions.
  :type action_dim: int
  :param n_agents: The number of all agents.
  :type n_agents: int
  :param hidden_sizes: The number of hidden units for hidden layers.
  :type hidden_sizes: list
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param gain: optional scaling factor.
  :type gain: float
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.categorical_mal.ActorNet.call(x)

  :param x: The input tensor.
  :type x: tf.Tensor
  :return: The logits for the policy.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.policies.categorical_mal.CriticNet(state_dim, n_agents, hidden_sizes,  normalize, initialize, activation, device)

  The critic network that is used to evaluate the states or state-action pairs.

  :param state_dim: The dimension of the state varibale.
  :type state_dim: int
  :param n_agents: number of agents.
  :type n_agents: int
  :param hidden_sizes: The number of hidden units for hidden layers.
  :type hidden_sizes: list
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.categorical_mal.CriticNet.call(x)

  :param x: input data.
  :type x: tf.Tensor
  :return: The evaluated critic value.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.policies.categorical_mal.COMA_Critic(state_dim, act_dim, hidden_sizes,  normalize, initialize, activation, device)

  The critic network for the Counterfactual Multi-Agent (COMA) algorithm, 
  which is used to evaluate the global state of the multi-agent systems.

  :param state_dim: The dimension of the state varibale.
  :type state_dim: int
  :param act_dim: The dimension of actions.
  :type act_dim: int
  :param hidden_sizes: The number of hidden units for hidden layers.
  :type hidden_sizes: list
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.categorical_mal.COMA_Critic.call(x)

  :param x: The input tensor.
  :type x: tf.Tensor
  :return: The evaluated critic value.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.policies.categorical_mal.MAAC_Policy(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  Multi-Agent Actor-Critic (MAAC) policy for discrete action space. It is used to generate categorical distributions and the total values of the multi-agent team.

  :param action_space: The action space.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: tk.Model
  :param mixer: The mixer for independent values.
  :type mixer: tk.Model
  :param actor_hidden_size: The number of hidden units for actor's hidden layers.
  :type actor_hidden_size: list
  :param critic_hidden_size: The number of hidden units for critic's hidden layers.
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
  xuance.tensorflow.policies.categorical_mal.MAAC_Policy.call(observation, agent_ids, *rnn_hidden, avail_actions=None)

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: tf.Tensor
  :return: A tuple that includes the final rnn hidden state, and the stochastic policies.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.categorical_mal.MAAC_Policy.get_values(critic_in, agent_ids, *rnn_hidden)

  :param critic_in: The input variables of critic networks.
  :type critic_in: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :return: A tuple that includes the final rnn hidden state, and the total values of the team.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.categorical_mal.MAAC_Policy.value_tot(values_n, global_state)

  Calculate the total values of the team via the value decomposition modules (self.mixer).

  :param values_n: The joint values of n agents.
  :type values_n: tf.Tensor
  :param global_state: The global states of the environments.
  :type global_state: tf.Tensor
  :return: The final rnn hidden state, and the total values of the team.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.categorical_mal.MAAC_Policy.trainable_param()

  Get the trainable parameters of the policy.

  :return: Trainable parameters.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.policies.categorical_mal.MAAC_Policy_Share(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: tk.Model
  :param mixer: The mixer for independent values.
  :type mixer: tk.Model
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
  xuance.tensorflow.policies.categorical_mal.MAAC_Policy_Share.call(observation, agent_ids, *rnn_hidden, avail_actions=None)

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: tf.Tensor
  :return: A tuple that includes the final rnn hidden state, the stochastic policies, and the total values of the team.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.categorical_mal.MAAC_Policy_Share.value_tot(values_n, global_state)

  :param values_n: The joint values of n agents.
  :type values_n: tf.Tensor
  :param global_state: The global states of the environments.
  :type global_state: tf.Tensor
  :return: A tuple that includes the final rnn hidden state, and the total values of the team.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.categorical_mal.MAAC_Policy_Share.trainable_param()

  Get the trainable parameters of the policy.

  :return: Trainable parameters.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.policies.categorical_mal.COMAPolicy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  A policy for the Counterfactual Multi-Agent Policy (COMA) algorithm.

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
  xuance.tensorflow.policies.categorical_mal.COMAPolicy.call(observation, agent_ids, *rnn_hidden, avail_actions=None)

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: tf.Tensor
  :return: A tuple that includes the final rnn hidden state, and the action probabilities.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.categorical_mal.COMAPolicy.get_values(critic_in, *rnn_hidden, target=False)

  :param critic_in: The input variables of critic networks.
  :type critic_in: tf.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :param target: Determine whether to use target networks to calculate the values.
  :param target: bool
  :return: The team values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.categorical_mal.COMAPolicy.param_actor()

  Get the parameters of the actor netwroks and the corresponding representation networks (if exists).

  :return: Trainable parameters of the actor netwroks and the corresponding representation networks.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.categorical_mal.COMAPolicy.copy_target()

  Synchronize the target networks.

.. py:class::
  xuance.tensorflow.policies.categorical_mal.MeanFieldActorCriticPolicy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device, **kwargs)

  Mean Field Actor-Critic policy.

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
  :param kwargs: The other arguments.
  :type kwargs: dict

.. py:function::
  xuance.tensorflow.policies.categorical_mal.MeanFieldActorCriticPolicy.call(inputs)

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: A tuple of tensors that includes the outputs hidden states and the categorical distributions.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.categorical_mal.MeanFieldActorCriticPolicy.trainable_param()

  :return: Trainable parameters.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.categorical_mal.MeanFieldActorCriticPolicy.critic(observation, actions_mean, agent_ids)

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param actions_mean: The mean values of actions.
  :type actions_mean: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :return: The evaluated values of the observations and mean-actions pairs.
  :rtype: tf.Tensor

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.policies.categorical_marl.ActorNet(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation)

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
  xuance.mindspore.policies.categorical_marl.ActorNet.construct(x)

  The actor network that is used to calculate the logits of categorical distributions.

  :param x: The input tensor.
  :type x: ms.Tensor
  :return: The logits for the policy.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.categorical_marl.CriticNet(state_dim, n_agents, hidden_sizes, normalize, initialize, activation)

  The critic network that is used to evaluate the states or state-action pairs.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.mindspore.policies.categorical_marl.CriticNet.construct(x)

  :param x: The input tensor.
  :type x: ms.Tensor
  :return: The evaluated critic value.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.categorical_marl.COMA_Critic(state_dim, act_dim, hidden_sizes, normalize, initialize, activation)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param act_dim: The dimension of actions.
  :type act_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell

.. py:function::
  xuance.mindspore.policies.categorical_marl.COMA_Critic.construct(x)

  :param x: The input tensor.
  :type x: ms.Tensor
  :return: The evaluated critic value.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.categorical_marl.MAAC_Policy(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  Multi-Agent Actor-Critic (MAAC) policy for discrete action space. 
  It is used to generate categorical distributions and the total values of the multi-agent team.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Cell
  :param mixer: The mixer for independent values.
  :type mixer: nn.Cell
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
  xuance.mindspore.policies.categorical_marl.MAAC_Policy.construct(observation, agent_ids, rnn_hidden, avail_actions)

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: ms.Tensor
  :return: A tuple that includes the final rnn hidden state, and the stochastic policies.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.categorical_marl.MAAC_Policy.get_values(critic_in, agent_ids, rnn_hidden)

  Get the critic values of the agents.

  :param critic_in: The input variables of critic networks.
  :type critic_in: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :return: A tuple that includes the final rnn hidden state, and the total values of the team.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.categorical_marl.MAAC_Policy.value_tot(values_n, global_state)

  Calculate the total values of the team via the value decomposition modules (self.mixer).

  :param values_n: The joint values of n agents.
  :type values_n: ms.Tensor
  :param global_state: The global states of the environments.
  :type global_state: ms.Tensor
  :return: The final rnn hidden state, and the total values of the team.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.categorical_marl.MAAC_Policy_Share(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)
  
  Similar to MAAC_Policy but shares representations between agents.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Cell
  :param mixer: The mixer for independent values.
  :type mixer: nn.Cell
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
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.mindspore.policies.categorical_marl.MAAC_Policy_Share.construct(observation, agent_ids, rnn_hidden, avail_actions)

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: ms.Tensor
  :return: A tuple that includes the final rnn hidden state, the stochastic policies, and the total values of the team.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.categorical_marl.MAAC_Policy_Share.value_tot(values_n, global_state)

  Calculate the total values of the team via the value decomposition modules (self.mixer).

  :param values_n: The joint values of n agents.
  :type values_n: ms.Tensor
  :param global_state: The global states of the environments.
  :type global_state: ms.Tensor
  :return: A tuple that includes the final rnn hidden state, and the total values of the team.
  :rtype: tuple

.. py:class::
  xuance.mindspore.policies.categorical_marl.COMAPolicy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  A policy for the Counterfactual Multi-Agent Policy (COMA) algorithm.

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
  xuance.mindspore.policies.categorical_marl.COMAPolicy.construct(observation, agent_ids, rnn_hidden, avail_actions, epsilon)

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: ms.Tensor
  :param epsilon: Greedy factor to select actions.
  :type epsilon: float
  :return: A tuple that includes the final rnn hidden state, and the action probabilities.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.categorical_marl.COMAPolicy.get_values(critic_in, rnn_hidden, target)

  :param critic_in: The input variables of critic networks.
  :type critic_in: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :param target: Determine whether to use target networks to calculate the values.
  :param target: bool
  :return: The team values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.categorical_marl.COMAPolicy.copy_target()

  Synchronize the target networks.

.. py:class::
  xuance.mindspore.policies.categorical_marl.MeanFieldActorCriticPolicy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  Mean Field Actor-Critic policy.

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
  xuance.mindspore.policies.categorical_marl.MeanFieldActorCriticPolicy.construct(observation, agent_ids)

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: A tuple of tensors that includes the outputs hidden states and the categorical distributions.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.categorical_marl.MeanFieldActorCriticPolicy.get_values(observation, actions_mean, agent_ids)

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param actions_mean: The mean values of actions.
  :type actions_mean: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The evaluated values of the observations and mean-actions pairs.
  :rtype: ms.Tensor

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        import torch

        from xuance.torch.policies import *
        from xuance.torch.utils import *
        from xuance.torch.representations import Basic_Identical
        from .deterministic_marl import BasicQhead


        class ActorNet(nn.Module):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         n_agents: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         gain: float = 1.0,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(ActorNet, self).__init__()
                layers = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize,
                                                 device=device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, None, initialize, device)[0])
                self.pi_logits = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor):
                return self.pi_logits(x)


        class CriticNet(nn.Module):
            def __init__(self,
                         state_dim: int,
                         n_agents: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(CriticNet, self).__init__()
                layers = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device=device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device=device)[0])
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor):
                return self.model(x)


        class COMA_Critic(nn.Module):
            def __init__(self,
                         state_dim: int,
                         act_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(COMA_Critic, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], act_dim, None, None, None, device)[0])
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor):
                return self.model(x)


        class MAAC_Policy(nn.Module):
            """
            MAAC_Policy: Multi-Agent Actor-Critic Policy
            """

            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: nn.Module,
                         mixer: Optional[VDN_mixer] = None,
                         actor_hidden_size: Sequence[int] = None,
                         critic_hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None,
                         **kwargs):
                super(MAAC_Policy, self).__init__()
                self.device = device
                self.action_dim = action_space.n
                self.n_agents = n_agents
                self.representation = representation[0]
                self.representation_critic = representation[1]
                self.representation_info_shape = self.representation.output_shapes
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.actor = ActorNet(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                      actor_hidden_size, normalize, initialize, kwargs['gain'], activation, device)
                self.critic = CriticNet(self.representation_critic.output_shapes['state'][0], n_agents, critic_hidden_size,
                                        normalize, initialize, activation, device)
                self.mixer = mixer
                self.pi_dist = CategoricalDistribution(self.action_dim)

            def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                        *rnn_hidden: torch.Tensor, avail_actions=None):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                actor_input = torch.concat([outputs['state'], agent_ids], dim=-1)
                act_logits = self.actor(actor_input)
                if avail_actions is not None:
                    avail_actions = torch.Tensor(avail_actions)
                    act_logits[avail_actions == 0] = -1e10
                    self.pi_dist.set_param(logits=act_logits)
                else:
                    self.pi_dist.set_param(logits=act_logits)
                return rnn_hidden, self.pi_dist

            def get_values(self, critic_in: torch.Tensor, agent_ids: torch.Tensor, *rnn_hidden: torch.Tensor):
                shape_obs = critic_in.shape
                # get representation features
                if self.use_rnn:
                    batch_size, n_agent, episode_length, dim_obs = tuple(shape_obs)
                    outputs = self.representation_critic(critic_in.reshape(-1, episode_length, dim_obs), *rnn_hidden)
                    outputs['state'] = outputs['state'].view(batch_size, n_agent, episode_length, -1)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    batch_size, n_agent, dim_obs = tuple(shape_obs)
                    outputs = self.representation_critic(critic_in.reshape(-1, dim_obs))
                    outputs['state'] = outputs['state'].view(batch_size, n_agent, -1)
                    rnn_hidden = None
                # get critic values
                critic_in = torch.concat([outputs['state'], agent_ids], dim=-1)
                v = self.critic(critic_in)
                return rnn_hidden, v

            def value_tot(self, values_n: torch.Tensor, global_state=None):
                if global_state is not None:
                    global_state = torch.as_tensor(global_state).to(self.device)
                return values_n if self.mixer is None else self.mixer(values_n, global_state)


        class MAAC_Policy_Share(MAAC_Policy):
            """
            MAAC_Policy: Multi-Agent Actor-Critic Policy
            """

            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: nn.Module,
                         mixer: Optional[VDN_mixer] = None,
                         actor_hidden_size: Sequence[int] = None,
                         critic_hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None,
                         **kwargs):
                super(MAAC_Policy, self).__init__()
                self.device = device
                self.action_dim = action_space.n
                self.n_agents = n_agents
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                      actor_hidden_size, normalize, initialize, kwargs['gain'], activation, device)
                self.critic = CriticNet(self.representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                        normalize, initialize, activation, device)
                self.mixer = mixer
                self.pi_dist = CategoricalDistribution(self.action_dim)

            def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                        *rnn_hidden: torch.Tensor, avail_actions=None, state=None):
                batch_size = len(avail_actions)
                if self.use_rnn:
                    sequence_length = observation.shape[1]
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                    representated_state = outputs['state'].view(batch_size, self.n_agents, sequence_length, -1)
                    actor_critic_input = torch.concat([representated_state, agent_ids], dim=-1)
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                    actor_critic_input = torch.concat([outputs['state'], agent_ids], dim=-1)
                act_logits = self.actor(actor_critic_input)
                if avail_actions is not None:
                    avail_actions = torch.Tensor(avail_actions)
                    act_logits[avail_actions == 0] = -1e10
                    self.pi_dist.set_param(logits=act_logits)
                else:
                    self.pi_dist.set_param(logits=act_logits)

                values_independent = self.critic(actor_critic_input)
                if self.use_rnn:
                    if self.mixer is None:
                        values_tot = values_independent
                    else:
                        sequence_length = observation.shape[1]
                        values_independent = values_independent.transpose(1, 2).reshape(batch_size*sequence_length, self.n_agents)
                        values_tot = self.value_tot(values_independent, global_state=state)
                        values_tot = values_tot.reshape([batch_size, sequence_length, 1])
                        values_tot = values_tot.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                else:
                    values_tot = values_independent if self.mixer is None else self.value_tot(values_independent, global_state=state)

                return rnn_hidden, self.pi_dist, values_tot

            def value_tot(self, values_n: torch.Tensor, global_state=None):
                if global_state is not None:
                    global_state = torch.as_tensor(global_state).to(self.device)
                return values_n if self.mixer is None else self.mixer(values_n, global_state)


        class COMAPolicy(nn.Module):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: nn.Module,
                         actor_hidden_size: Sequence[int] = None,
                         critic_hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None,
                         **kwargs):
                super(COMAPolicy, self).__init__()
                self.device = device
                self.action_dim = action_space.n
                self.n_agents = n_agents
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.actor = ActorNet(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                      actor_hidden_size, normalize, initialize, kwargs['gain'], activation, device)
                critic_input_dim = self.representation.input_shape[0] + self.action_dim * self.n_agents
                if kwargs["use_global_state"]:
                    critic_input_dim += kwargs["dim_state"]
                self.critic = COMA_Critic(critic_input_dim, self.action_dim, critic_hidden_size,
                                          normalize, initialize, activation, device)
                self.target_critic = copy.deepcopy(self.critic)
                self.parameters_critic = list(self.critic.parameters())
                self.parameters_actor = list(self.representation.parameters()) + list(self.actor.parameters())
                self.pi_dist = CategoricalDistribution(self.action_dim)

            def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                        *rnn_hidden: torch.Tensor, avail_actions=None, epsilon=0.0):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                actor_input = torch.concat([outputs['state'], agent_ids], dim=-1)
                act_logits = self.actor(actor_input)
                act_probs = nn.functional.softmax(act_logits, dim=-1)
                act_probs = (1 - epsilon) * act_probs + epsilon * 1 / self.action_dim
                if avail_actions is not None:
                    avail_actions = torch.Tensor(avail_actions)
                    act_probs[avail_actions == 0] = 0.0
                return rnn_hidden, act_probs

            def get_values(self, critic_in: torch.Tensor, *rnn_hidden: torch.Tensor, target=False):
                # get critic values
                v = self.target_critic(critic_in) if target else self.critic(critic_in)
                return [None, None], v

            def copy_target(self):
                for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                    tp.data.copy_(ep)


        class MeanFieldActorCriticPolicy(nn.Module):
            def __init__(self,
                         action_space: Discrete,
                         n_agents: int,
                         representation: nn.Module,
                         actor_hidden_size: Sequence[int] = None,
                         critic_hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None
                         ):
                super(MeanFieldActorCriticPolicy, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.actor_net = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                          actor_hidden_size, normalize, initialize, activation, device)
                self.critic_net = BasicQhead(representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                             n_agents, critic_hidden_size, normalize, initialize, activation, device)
                self.target_actor_net = copy.deepcopy(self.actor_net)
                self.target_critic_net = copy.deepcopy(self.critic_net)
                self.parameters_actor = list(self.actor_net.parameters()) + list(self.representation.parameters())
                self.parameters_critic = self.critic_net.parameters()

            def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
                outputs = self.representation(observation)
                input_actor = torch.concat([outputs['state'], agent_ids], dim=-1)
                act_dist = self.actor_net(input_actor)
                return outputs, act_dist

            def target_actor(self, observation: torch.Tensor, agent_ids: torch.Tensor):
                outputs = self.representation(observation)
                input_actor = torch.concat([outputs['state'], agent_ids], dim=-1)
                act_dist = self.target_actor_net(input_actor)
                return act_dist

            def critic(self, observation: torch.Tensor, actions_mean: torch.Tensor, agent_ids: torch.Tensor):
                outputs = self.representation(observation)
                critic_in = torch.concat([outputs['state'], actions_mean, agent_ids], dim=-1)
                return self.critic_net(critic_in)

            def target_critic(self, observation: torch.Tensor, actions_mean: torch.Tensor, agent_ids: torch.Tensor):
                outputs = self.representation(observation)
                critic_in = torch.concat([outputs['state'], actions_mean, agent_ids], dim=-1)
                return self.target_critic_net(critic_in)

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor_net.parameters(), self.target_actor_net.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.critic_net.parameters(), self.target_critic_net.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)


  .. group-tab:: TensorFlow

    .. code-block:: python

        from xuance.tensorflow.policies import *
        from xuance.tensorflow.utils import *
        from xuance.tensorflow.representations import Basic_Identical


        class ActorNet(tk.Model):
            def __init__(self,
                        state_dim: int,
                        action_dim: int,
                        n_agents: int,
                        hidden_sizes: Sequence[int],
                        normalize: Optional[tk.layers.Layer] = None,
                        initializer: Optional[tk.initializers.Initializer] = None,
                        gain: float = 1.0,
                        activation: Optional[tk.layers.Layer] = None,
                        device: str = "cpu:0"):
                super(ActorNet, self).__init__()
                layers = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, None, initializer, device=device)[0])
                self.pi_logits = tk.Sequential(layers)
                self.dist = CategoricalDistribution(action_dim)

            def call(self, x: tf.Tensor, **kwargs):
                self.dist.set_param(self.pi_logits(x))
                return self.pi_logits(x)


        class CriticNet(tk.Model):
            def __init__(self,
                        state_dim: int,
                        n_agents: int,
                        hidden_sizes: Sequence[int],
                        normalize: Optional[tk.layers.Layer] = None,
                        initializer: Optional[tk.initializers.Initializer] = None,
                        activation: Optional[tk.layers.Layer] = None,
                        device: Optional[Union[str, int]] = None):
                super(CriticNet, self).__init__()
                layers = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
                self.model = tk.Sequential(layers)

            def call(self, x: tf.Tensor, **kwargs):
                return self.model(x)[:, :, 0]


        class COMA_CriticNet(tk.Model):
            def __init__(self,
                        state_dim: int,
                        act_dim: int,
                        hidden_sizes: Sequence[int],
                        normalize: Optional[tk.layers.Layer] = None,
                        initializer: Optional[tk.initializers.Initializer] = None,
                        activation: Optional[tk.layers.Layer] = None,
                        device: Optional[Union[str, int]] = None):
                super(COMA_CriticNet, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], act_dim, None, None, None, device)[0])
                self.model = tk.Sequential(layers)

            def call(self, x: tf.Tensor, **kwargs):
                return self.model(x)


        class MAAC_Policy(tk.Model):
            """
            MAAC_Policy: Multi-Agent Actor-Critic Policy
            """
            def __init__(self,
                        action_space: Discrete,
                        n_agents: int,
                        representation: Optional[Basic_Identical],
                        mixer: Optional[VDN_mixer] = None,
                        actor_hidden_size: Sequence[int] = None,
                        critic_hidden_size: Sequence[int] = None,
                        normalize: Optional[tk.layers.Layer] = None,
                        initializer: Optional[tk.initializers.Initializer] = None,
                        activation: Optional[tk.layers.Layer] = None,
                        device: Optional[Union[str, int]] = None,
                        **kwargs):
                super(MAAC_Policy, self).__init__()
                self.device = device
                self.action_dim = action_space.n
                self.n_agents = n_agents
                self.representation = representation[0]
                self.representation_critic = representation[1]
                self.representation_info_shape = self.representation.output_shapes
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.actor = ActorNet(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                    actor_hidden_size, normalize, initializer, kwargs['gain'], activation, device)
                self.critic = CriticNet(self.representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                        normalize, initializer, activation, device)
                self.mixer = mixer
                self.identical_rep = True if isinstance(self.representation, Basic_Identical) else False
                self.pi_dist = CategoricalDistribution(self.action_dim)

            def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
                observation = inputs['obs']
                agent_ids = inputs['ids']
                obs_shape = observation.shape
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    outputs_state = outputs['state']  # need to be improved
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    observation_reshape = tf.reshape(observation, [-1, obs_shape[-1]])
                    outputs = self.representation(observation_reshape)
                    outputs_state = tf.reshape(outputs['state'], obs_shape[:-1] + self.representation_info_shape['state'])
                    rnn_hidden = None
                actor_input = tf.concat([outputs_state, agent_ids], axis=-1)
                act_logits = self.actor(actor_input)
                if ('avail_actions' in kwargs.keys()) and (kwargs['avail_actions'] is not None):
                    avail_actions = tf.convert_to_tensor(kwargs['avail_actions'])
                    act_logits[avail_actions == 0] = -1e10
                    self.pi_dist.set_param(logits=act_logits)
                else:
                    self.pi_dist.set_param(logits=act_logits)
                return rnn_hidden, self.pi_dist

            def get_values(self, critic_in: tf.Tensor, agent_ids: tf.Tensor, *rnn_hidden: tf.Tensor):
                shape_obs = critic_in.shape
                # get representation features
                if self.use_rnn:
                    batch_size, n_agent, episode_length, dim_obs = tuple(shape_obs)
                    outputs = self.representation_critic(critic_in.reshape(-1, episode_length, dim_obs), *rnn_hidden)
                    outputs['state'] = outputs['state'].view(batch_size, n_agent, episode_length, -1)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    batch_size, n_agent, dim_obs = tuple(shape_obs)
                    outputs = self.representation_critic(tf.reshape(critic_in, [-1, dim_obs]))
                    outputs['state'] = tf.reshape(outputs['state'], [batch_size, n_agent, -1])
                    rnn_hidden = None
                # get critic values
                critic_in = tf.concat([outputs['state'], agent_ids], axis=-1)
                v = self.critic(critic_in)
                return rnn_hidden, v

            def value_tot(self, values_n: tf.Tensor, global_state=None):
                if global_state is not None:
                    with tf.device(self.device):
                        global_state = tf.convert_to_tensor(global_state)
                return values_n if self.mixer is None else self.mixer(values_n, global_state)

            def trainable_param(self):
                params = self.actor.trainable_variables + self.critic.trainable_variables
                if self.mixer is not None:
                    params += self.mixer.trainable_variables
                if self.identical_rep:
                    return params
                else:
                    return params + self.representation.trainable_variables + self.representation_critic.trainable_variables


        class MAAC_Policy_Share(MAAC_Policy):
            def __init__(self,
                        action_space: Discrete,
                        n_agents: int,
                        representation: tk.Model,
                        mixer: Optional[VDN_mixer] = None,
                        actor_hidden_size: Sequence[int] = None,
                        critic_hidden_size: Sequence[int] = None,
                        normalize: Optional[tk.layers.Layer] = None,
                        initialize: Optional[tk.initializers.Initializer] = None,
                        activation: Optional[tk.layers.Layer] = None,
                        device: Optional[Union[str, int]] = None,
                        **kwargs):
                super(MAAC_Policy, self).__init__()
                self.device = device
                self.action_dim = action_space.n
                self.n_agents = n_agents
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                    actor_hidden_size, normalize, initialize, kwargs['gain'], activation, device)
                self.critic = CriticNet(self.representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                        normalize, initialize, activation, device)
                self.mixer = mixer
                self.identical_rep = True if isinstance(self.representation, Basic_Identical) else False
                self.pi_dist = CategoricalDistribution(self.action_dim)

            def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
                observation = inputs['obs']
                agent_ids = inputs['ids']
                obs_shape = observation.shape
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    outputs_state = outputs['state']  # need to be improved
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    observation_reshape = tf.reshape(observation, [-1, obs_shape[-1]])
                    outputs = self.representation(observation_reshape)
                    outputs_state = tf.reshape(outputs['state'], obs_shape[:-1] + self.representation_info_shape['state'])
                    rnn_hidden = None
                actor_critic_input = tf.concat([outputs_state, agent_ids], axis=-1)
                act_logits = self.actor(actor_critic_input)
                if ('avail_actions' in kwargs.keys()) and (kwargs['avail_actions'] is not None):
                    avail_actions = tf.convert_to_tensor(kwargs['avail_actions'])
                    act_logits[avail_actions == 0] = -1e10
                    self.pi_dist.set_param(logits=act_logits)
                else:
                    self.pi_dist.set_param(logits=act_logits)

                values_independent = self.critic(actor_critic_input)
                if self.use_rnn:
                    pass  # to do
                else:
                    values_tot = values_independent if self.mixer is None else self.value_tot(values_independent,
                                                                                            global_state=kwargs['state'])
                    values_tot = tf.repeat(tf.expand_dims(values_tot, 1), repeats=self.n_agents, axis=1)

                return rnn_hidden, self.pi_dist, values_tot

            def value_tot(self, values_n: tf.Tensor, global_state=None):
                if global_state is not None:
                    with tf.device(self.device):
                        global_state = tf.convert_to_tensor(global_state)
                return values_n if self.mixer is None else self.mixer(values_n, global_state)

            def trainable_param(self):
                params = self.actor.trainable_variables + self.critic.trainable_variables
                if self.mixer is not None:
                    params += self.mixer.trainable_variables
                if self.identical_rep:
                    return params
                else:
                    return params + self.representation.trainable_variables


        class COMAPolicy(tk.Model):
            def __init__(self,
                        action_space: Discrete,
                        n_agents: int,
                        representation: Optional[Basic_Identical],
                        actor_hidden_size: Sequence[int] = None,
                        critic_hidden_size: Sequence[int] = None,
                        normalize: Optional[tk.layers.Layer] = None,
                        initializer: Optional[tk.initializers.Initializer] = None,
                        activation: Optional[tk.layers.Layer] = None,
                        device: Optional[Union[str, int]] = None,
                        **kwargs):
                super(COMAPolicy, self).__init__()
                self.device = device
                self.action_dim = action_space.n
                self.n_agents = n_agents
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                    actor_hidden_size, normalize, initializer, kwargs['gain'], activation, device)
                critic_input_dim = kwargs['dim_obs'] + self.action_dim * self.n_agents
                if kwargs["use_global_state"]:
                    critic_input_dim += kwargs["dim_state"]
                self.critic = COMA_CriticNet(critic_input_dim, self.action_dim, critic_hidden_size,
                                            normalize, initializer, activation, device)
                self.target_critic = COMA_CriticNet(critic_input_dim, self.action_dim, critic_hidden_size,
                                                    normalize, initializer, activation, device)
                self.parameters_critic = self.critic.trainable_variables
                self.pi_dist = CategoricalDistribution(self.action_dim)

            def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
                observation = inputs['obs']
                agent_ids = inputs['ids']
                obs_shape = observation.shape
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    outputs_state = outputs['state']  # need to be improved
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    observation_reshape = tf.reshape(observation, [-1, obs_shape[-1]])
                    outputs = self.representation(observation_reshape)
                    outputs_state = tf.reshape(outputs['state'], obs_shape[:-1] + self.representation_info_shape['state'])
                    rnn_hidden = None
                actor_input = tf.concat([outputs_state, agent_ids], axis=-1)
                act_logits = self.actor(actor_input)
                act_probs = tf.nn.softmax(act_logits, axis=-1)
                act_probs = (1 - kwargs['epsilon']) * act_probs + kwargs['epsilon'] * 1 / self.action_dim
                if ('avail_actions' in kwargs.keys()) and (kwargs['avail_actions'] is not None):
                    avail_actions = tf.Tensor(kwargs['avail_actions'])
                    act_probs[avail_actions == 0] = 0.0
                return rnn_hidden, act_probs

            def get_values(self, critic_in: tf.Tensor, *rnn_hidden: tf.Tensor, target=False):
                # get critic values
                v = self.target_critic(critic_in) if target else self.critic(critic_in)
                return [None, None], v

            def param_actor(self):
                if isinstance(self.representation, Basic_Identical):
                    return self.actor.trainable_variables
                else:
                    return self.representation.trainable_variables + self.actor.trainable_variables

            def copy_target(self):
                self.target_critic.set_weights(self.critic.get_weights())


        class MeanFieldActorCriticPolicy(tk.Model):
            def __init__(self,
                        action_space: Discrete,
                        n_agents: int,
                        representation: tk.Model,
                        actor_hidden_size: Sequence[int] = None,
                        critic_hidden_size: Sequence[int] = None,
                        normalize: Optional[tk.layers.Layer] = None,
                        initializer: Optional[tk.initializers.Initializer] = None,
                        activation: Optional[tk.layers.Layer] = None,
                        device: Optional[Union[str, int]] = None,
                        **kwargs):
                super(MeanFieldActorCriticPolicy, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.actor_net = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                        actor_hidden_size, normalize, initializer, kwargs['gain'], activation, device)
                self.critic_net = CriticNet(representation.output_shapes['state'][0] + self.action_dim, n_agents,
                                            critic_hidden_size, normalize, initializer, activation, device)
                self.trainable_param = self.actor_net.trainable_variables + self.critic_net.trainable_variables
                self.identical_rep = True if isinstance(self.representation, Basic_Identical) else False
                self.pi_dist = CategoricalDistribution(self.action_dim)

            def call(self, inputs: Union[np.ndarray, dict], **kwargs):
                observations = inputs['obs']
                IDs = inputs['ids']
                outputs = self.representation(observations)
                input_actor = tf.concat([outputs['state'], IDs], axis=-1)
                act_logits = self.actor_net(input_actor)
                self.pi_dist.set_param(logits=act_logits)
                return outputs, self.pi_dist

            def trainable_param(self):
                params = self.actor_net.trainable_variables + self.critic_net.trainable_variables
                if self.identical_rep:
                    return params
                else:
                    return params + self.representation.trainable_variables

            def critic(self, observation: tf.Tensor, actions_mean: tf.Tensor, agent_ids: tf.Tensor):
                outputs = self.representation(observation)
                critic_in = tf.concat([outputs['state'], actions_mean, agent_ids], axis=-1)
                critic_out = tf.expand_dims(self.critic_net(critic_in), -1)
                return critic_out



  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.policies import *
        from xuance.mindspore.utils import *
        from xuance.mindspore.representations import Basic_Identical
        from mindspore.nn.probability.distribution import Categorical
        import copy


        class ActorNet(nn.Cell):
            class Sample(nn.Cell):
                def __init__(self):
                    super(ActorNet.Sample, self).__init__()
                    self._dist = Categorical(dtype=ms.float32)

                def construct(self, probs: ms.tensor):
                    return self._dist.sample(probs=probs).astype("int32")

            class LogProb(nn.Cell):
                def __init__(self):
                    super(ActorNet.LogProb, self).__init__()
                    self._dist = Categorical(dtype=ms.float32)

                def construct(self, value, probs):
                    return self._dist._log_prob(value=value, probs=probs)

            class Entropy(nn.Cell):
                def __init__(self):
                    super(ActorNet.Entropy, self).__init__()
                    self._dist = Categorical(dtype=ms.float32)

                def construct(self, probs):
                    return self._dist.entropy(probs=probs)

            class KL_Div(nn.Cell):
                def __init__(self):
                    super(ActorNet.KL_Div, self).__init__()
                    self._dist = Categorical(dtype=ms.float32)

                def construct(self, probs_p, probs_q):
                    return self._dist.kl_loss('Categorical', probs_p, probs_q)

            def __init__(self,
                        state_dim: int,
                        action_dim: int,
                        n_agents: int,
                        hidden_sizes: Sequence[int],
                        normalize: Optional[ModuleType] = None,
                        initialize: Optional[Callable[..., ms.Tensor]] = None,
                        gain: float = 1.0,
                        activation: Optional[ModuleType] = None):
                super(ActorNet, self).__init__()
                layers = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, None, initialize)[0])
                self.model = nn.SequentialCell(*layers)
                self.sample = self.Sample()
                self.log_prob = self.LogProb()
                self.entropy = self.Entropy()
                self.kl_div = self.KL_Div()

            def construct(self, x: ms.Tensor):
                return self.model(x)


        class CriticNet(nn.Cell):
            def __init__(self,
                        state_dim: int,
                        n_agents: int,
                        hidden_sizes: Sequence[int],
                        normalize: Optional[ModuleType] = None,
                        initialize: Optional[Callable[..., ms.Tensor]] = None,
                        activation: Optional[ModuleType] = None):
                super(CriticNet, self).__init__()
                layers = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, None)[0])
                self.model = nn.SequentialCell(*layers)

            def construct(self, x: ms.Tensor):
                return self.model(x)


        class COMA_Critic(nn.Cell):
            def __init__(self,
                        state_dim: int,
                        act_dim: int,
                        hidden_sizes: Sequence[int],
                        normalize: Optional[ModuleType] = None,
                        initialize: Optional[Callable[..., ms.Tensor]] = None,
                        activation: Optional[ModuleType] = None):
                super(COMA_Critic, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], act_dim, None, None, None)[0])
                self.model = nn.SequentialCell(*layers)

            def construct(self, x: ms.Tensor):
                return self.model(x)


        class MAAC_Policy(nn.Cell):
            def __init__(self,
                        action_space: Discrete,
                        n_agents: int,
                        representation: Optional[Basic_Identical],
                        mixer: Optional[VDN_mixer] = None,
                        actor_hidden_size: Sequence[int] = None,
                        critic_hidden_size: Sequence[int] = None,
                        normalize: Optional[ModuleType] = None,
                        initialize: Optional[Callable[..., ms.Tensor]] = None,
                        activation: Optional[ModuleType] = None,
                        **kwargs):
                super(MAAC_Policy, self).__init__()
                self.action_dim = action_space.n
                self.n_agents = n_agents
                self.representation = representation[0]
                self.representation_critic = representation[1]
                self.representation_info_shape = self.representation.output_shapes
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.actor = ActorNet(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                    actor_hidden_size, normalize, initialize, kwargs['gain'], activation)
                self.critic = CriticNet(self.representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                        normalize, initialize, activation)
                self.mixer = mixer
                self._concat = ms.ops.Concat(axis=-1)
                self.expand_dims = ms.ops.ExpandDims()
                self._softmax = nn.Softmax(axis=-1)

            def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                        *rnn_hidden: ms.Tensor, avail_actions=None):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                actor_input = self._concat([outputs['state'], agent_ids])
                act_logits = self.actor(actor_input)
                if avail_actions is not None:
                    act_logits[avail_actions == 0] = -1e10
                    act_probs = self._softmax(act_logits)
                else:
                    act_probs = self._softmax(act_logits)
                return rnn_hidden, act_probs

            def get_values(self, critic_in: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: ms.Tensor):
                shape_obs = critic_in.shape
                # get representation features
                if self.use_rnn:
                    batch_size, n_agent, episode_length, dim_obs = tuple(shape_obs)
                    outputs = self.representation_critic(critic_in.reshape(-1, episode_length, dim_obs), *rnn_hidden)
                    outputs['state'] = outputs['state'].view(batch_size, n_agent, episode_length, -1)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    batch_size, n_agent, dim_obs = tuple(shape_obs)
                    outputs = self.representation_critic(critic_in.reshape(-1, dim_obs))
                    outputs['state'] = outputs['state'].view(batch_size, n_agent, -1)
                    rnn_hidden = None
                # get critic values
                critic_in = self._concat([outputs['state'], agent_ids])
                v = self.critic(critic_in)
                return rnn_hidden, v

            def value_tot(self, values_n: ms.Tensor, global_state=None):
                if global_state is not None:
                    global_state = global_state
                return values_n if self.mixer is None else self.mixer(values_n, global_state)


        class MAAC_Policy_Share(MAAC_Policy):
            """
            MAAC_Policy: Multi-Agent Actor-Critic Policy
            """

            def __init__(self,
                        action_space: Discrete,
                        n_agents: int,
                        representation: nn.Cell,
                        mixer: Optional[VDN_mixer] = None,
                        actor_hidden_size: Sequence[int] = None,
                        critic_hidden_size: Sequence[int] = None,
                        normalize: Optional[ModuleType] = None,
                        initialize: Optional[Callable[..., ms.Tensor]] = None,
                        activation: Optional[ModuleType] = None,
                        device: Optional[Union[str, int]] = None,
                        **kwargs):
                super(MAAC_Policy, self).__init__()
                self.device = device
                self.action_dim = action_space.n
                self.n_agents = n_agents
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                    actor_hidden_size, normalize, initialize, kwargs['gain'], activation)
                self.critic = CriticNet(self.representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                        normalize, initialize, activation)
                self.mixer = mixer
                self._concat = ms.ops.Concat(axis=-1)
                self.expand_dims = ms.ops.ExpandDims()
                self._softmax = nn.Softmax(axis=-1)

            def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                        *rnn_hidden: ms.Tensor, avail_actions=None, state=None):
                batch_size = len(observation)
                if self.use_rnn:
                    sequence_length = observation.shape[1]
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                    representated_state = outputs['state'].view(batch_size, self.n_agents, sequence_length, -1)
                    actor_critic_input = self._concat([representated_state, agent_ids])
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                    actor_critic_input = self._concat([outputs['state'], agent_ids])
                act_logits = self.actor(actor_critic_input)
                if avail_actions is not None:
                    act_logits[avail_actions == 0] = -1e10
                    act_probs = self._softmax(act_logits)
                else:
                    act_probs = self._softmax(act_logits)

                values_independent = self.critic(actor_critic_input)
                if self.use_rnn:
                    if self.mixer is None:
                        values_tot = values_independent
                    else:
                        sequence_length = observation.shape[1]
                        values_independent = values_independent.transpose(1, 2).reshape(batch_size * sequence_length,
                                                                                        self.n_agents)
                        values_tot = self.value_tot(values_independent, global_state=state)
                        values_tot = values_tot.reshape([batch_size, sequence_length, 1])
                        values_tot = values_tot.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                else:
                    values_tot = values_independent if self.mixer is None else self.value_tot(values_independent,
                                                                                            global_state=state)
                    values_tot = ms.ops.broadcast_to(values_tot.unsqueeze(1), (-1, self.n_agents, -1))

                return rnn_hidden, act_probs, values_tot

            def value_tot(self, values_n: ms.Tensor, global_state=None):
                if global_state is not None:
                    global_state = ms.Tensor(global_state)
                return values_n if self.mixer is None else self.mixer(values_n, global_state)


        class COMAPolicy(nn.Cell):
            def __init__(self,
                        action_space: Discrete,
                        n_agents: int,
                        representation: Optional[Basic_Identical],
                        actor_hidden_size: Sequence[int] = None,
                        critic_hidden_size: Sequence[int] = None,
                        normalize: Optional[ModuleType] = None,
                        initialize: Optional[Callable[..., ms.Tensor]] = None,
                        activation: Optional[ModuleType] = None,
                        **kwargs):
                super(COMAPolicy, self).__init__()
                self.action_dim = action_space.n
                self.n_agents = n_agents
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                    actor_hidden_size, normalize, initialize, kwargs['gain'], activation)
                critic_input_dim = self.representation.input_shape[0] + self.action_dim * self.n_agents
                if kwargs["use_global_state"]:
                    critic_input_dim += kwargs["dim_state"]
                self.critic = COMA_Critic(critic_input_dim, self.action_dim, critic_hidden_size,
                                        normalize, initialize, activation)
                self.target_critic = copy.deepcopy(self.critic)
                self.parameters_critic = self.critic.trainable_params()
                self.parameters_actor = self.representation.trainable_params() + self.actor.trainable_params()
                self.eye = ms.ops.Eye()
                self._softmax = nn.Softmax(axis=-1)
                self._concat = ms.ops.Concat(axis=-1)

            def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                        *rnn_hidden: ms.Tensor, avail_actions=None, epsilon=0.0):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                actor_input = self._concat([outputs['state'], agent_ids])
                act_logits = self.actor(actor_input)
                act_probs = self._softmax(act_logits)
                act_probs = (1 - epsilon) * act_probs + epsilon * 1 / self.action_dim
                if avail_actions is not None:
                    act_probs[avail_actions == 0] = 0.0
                return rnn_hidden, act_probs

            def get_values(self, critic_in: ms.Tensor, *rnn_hidden: ms.Tensor, target=False):
                # get critic values
                v = self.target_critic(critic_in) if target else self.critic(critic_in)
                return [None, None], v

            def copy_target(self):
                for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
                    tp.assign_value(ep)


        class MeanFieldActorCriticPolicy(nn.Cell):
            def __init__(self,
                        action_space: Discrete,
                        n_agents: int,
                        representation: Optional[Basic_Identical],
                        actor_hidden_size: Sequence[int] = None,
                        critic_hidden_size: Sequence[int] = None,
                        normalize: Optional[ModuleType] = None,
                        initialize: Optional[Callable[..., ms.Tensor]] = None,
                        activation: Optional[ModuleType] = None,
                        **kwargs):
                super(MeanFieldActorCriticPolicy, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                    actor_hidden_size, normalize, initialize, kwargs['gain'], activation)
                self.critic = CriticNet(representation.output_shapes['state'][0] + self.action_dim, n_agents,
                                        critic_hidden_size, normalize, initialize, activation)
                self.parameters_actor = self.actor.trainable_params() + self.representation.trainable_params()
                self.parameters_critic = self.critic.trainable_params()
                self._concat = ms.ops.Concat(axis=-1)

            def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor):
                outputs = self.representation(observation)
                input_actor = self._concat([outputs['state'], agent_ids])
                act_dist = self.actor(input_actor)
                return outputs, act_dist

            def get_values(self, observation: ms.Tensor, actions_mean: ms.Tensor, agent_ids: ms.Tensor):
                outputs = self.representation(observation)
                critic_in = self._concat([outputs['state'], actions_mean, agent_ids])
                return self.critic(critic_in)

