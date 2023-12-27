Gaussian-MARL
=======================================

In this module, we define several classes related to the components of the MARL policies with Gaussian distributions,
such as the Q-networks, actor networks, critic policies, and the policies.
These policies are used to generate actions from continuous aciton spaces, and calculate the values.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.policies.gaussian_marl.BasicQhead(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation, device)

  A basic module representing the Q-value head, commonly used in Q-networks. 
  It takes state, action, and agent information as input and produces Q-values as output.

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
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.gaussian_marl.BasicQhead.forward(x)

  Passes the input tensor x through the sequential model (self.model) and returns the output.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: The estimated Q-values.
  :rtype: torch.Tensor


.. py:class::
  xuance.torch.policies.gaussian_marl.BasicQnetwork(action_space, n_agents, representation, hidden_size, normalize, initialize, activation, device)

  A basic Q-network that utilizes the BasicQhead for both evaluation and target networks.

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
  xuance.torch.policies.gaussian_marl.BasicQnetwork.forward(observation, agent_ids)

  The forward method computes the evaluation Q-values for a given observation and agent identifiers, 
  along with other relevant information from the representation module.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: A tuple that includes the representation (outputs), argmax action (argmax_action), and evaluation Q-values (evalQ). These values can be useful for further processing during reinforcement learning training or evaluation.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.gaussian_marl.BasicQnetwork.target_Q(observation, agent_ids)

  The target_Q method computes the target Q-values for a given observation and agent identifiers using the target Q-head. 
  This method is typically used during the training process for updating the Q-network parameters based on the temporal difference error between the evaluation Q-values and the target Q-values.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: The target Q-values.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.gaussian_marl.BasicQnetwork.copy_target()

  Copies the parameters from the evaluation representation, target representation, evaluation Q-head, and target Q-head.


.. py:class::
  xuance.torch.policies.gaussian_marl.ActorNet(state_dim, n_agents, action_dim, hidden_sizes, normalize, initialize, activation, device)

  Represents the actor network, responsible for generating actions based on the given state and agent information. 
  It uses a Diagonal Gaussian distribution for the actions.

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
  xuance.torch.policies.gaussian_marl.ActorNet.forward(x)

  Passes the input tensor x through the sequential model (self.mu) to obtain the mean of the Gaussian distribution.
  Sets the parameters of the diagonal Gaussian distribution (self.dist) using the mean and the exponential of the log standard deviation.
  Returns the distribution object self.dist.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: The distribution object self.dist.

.. py:class::
  xuance.torch.policies.gaussian_marl.CriticNet(state_dim, n_agents, hidden_sizes, normalize, initialize, activation, device)

  Represents the critic network, which evaluates the state-action pairs.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param n_agents: The number of agents.
  :type n_agents: int
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
  xuance.torch.policies.gaussian_marl.CriticNet.forward(x)

  Passes the input tensor x through the sequential model (self.model) to obtain the output, 
  which represents the Q-values for the given state-action pairs.
  Returns the Q-values

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: The Q-values.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.gaussian_marl.MAAC_Policy(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  A multi-agent actor-critic policy with Gaussian policies. 
  It combines an actor network and a critic network and optionally uses a mixer to calculate the total team values.

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
  xuance.torch.policies.gaussian_marl.MAAC_Policy.forward(observation, agent_ids, *rnn_hidden)

  Depending on whether the policy uses RNN, the observation is passed through the representation network, and the hidden states are updated.
  The actor network is then applied to the concatenated input of agent states and IDs to obtain the probability distribution over actions (self.pi_dist).
  Returns the updated hidden states (if RNN is used) and the probability distribution.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :return: A tuple that includes the updated hidden states (if RNN is used) and the probability distribution.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.gaussian_marl.MAAC_Policy.get_values(critic_in, agent_ids, *rnn_hidden)

  Computes the critic values based on the input states, agent IDs, and optional RNN hidden states.

  :param critic_in: The input variables of critic networks.
  :type critic_in: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :return: The critic values.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.gaussian_marl.MAAC_Policy.value_tot(values_n, global_state)

  Computes the total team value, incorporating a mixer if provided.

  :param values_n: The joint values of n agents.
  :type values_n: torch.Tensor
  :param global_state: The global states of the environments.
  :type global_state: torch.Tensor
  :return: The total team value.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.gaussian_marl.Basic_ISAC_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  A basic policy architecture for the Independent Soft Actor-Critic (ISAC) algorithm, with independent actors and centralized critics. 
  It includes actor and critic networks, as well as target networks for stability during training. 
  The soft_update method is used to update the target networks gradually.

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
  xuance.torch.policies.gaussian_marl.Basic_ISAC_policy.forward(observation, agent_ids)

  Passes the observation through the agent representation network to obtain relevant features (outputs).
  Concatenates the state features with agent identifiers and passes them through the actor network to obtain actions (act).
  Returns the representation outputs and actions.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: A tuple that includes the representation outputs and actions.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.gaussian_marl.Basic_ISAC_policy.critic(observation, actions, agent_ids)

  Computes critic values for given observations, actions, and agent identifiers.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param actions: The actions input.
  :type actions: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: The evaluated critic values.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.gaussian_marl.Basic_ISAC_policy.target_critic(observation, actions, agent_ids)

  Computes critic values for target critic network given observations, actions, and agent identifiers.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param actions: The actions input.
  :type actions: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: The target critic values.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.gaussian_marl.Basic_ISAC_policy.target_actor(observation, agent_ids)

  Obtains the output of the target actor network.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: The output of the target actor network.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.gaussian_marl.Basic_ISAC_policy.soft_update(tau)

  Performs a soft update of the target networks using a parameter tau.
  Updates the target actor and target critic networks by blending their parameters with the corresponding parameters of the online networks.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  

.. py:class::
  xuance.torch.policies.gaussian_marl.MASAC_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  An extension of Basic_ISAC_policy for multi-agent environments. 
  It is an implementation of Multi-Agent Soft Actor-Critic (MASAC) algorithm.
  It includes modifications to the critic network to handle multiple agents.

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
  xuance.torch.policies.gaussian_marl.MASAC_policy.critic(observation, actions, agent_ids)

  Computes critic values for given observations, actions, and agent identifiers. 
  Reshapes the state and actions for multiple agents.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param actions: The actions input.
  :type actions: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: The evaluated critic values.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.gaussian_marl.MASAC_policy.target_critic(observation, actions, agent_ids)

  Computes critic values for the target critic network given observations, actions, and agent identifiers. 
  Reshapes the state and actions for multiple agents.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :param actions: The actions input.
  :type actions: torch.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: torch.Tensor
  :return: The target critic values.
  :rtype: torch.Tensor

.. raw:: html

    <br><hr>


TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.policies.gaussian_marl.BasicQhead(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation, device)

  A basic module representing the Q-value head, commonly used in Q-networks. 
  It takes state, action, and agent information as input and produces Q-values as output.

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
  xuance.tensorflow.policies.gaussian_marl.BasicQhead.call(x)

  Passes the input tensor x through the sequential model (self.model) and returns the output.

  :param x: The input tensor.
  :type x: tf.Tensor
  :return: The estimated Q-values.
  :rtype: tf.Tensor


.. py:class::
  xuance.tensorflow.policies.gaussian_marl.BasicQnetwork(action_space, n_agents, representation, hidden_size, normalize, initialize, activation, device)

  A basic Q-network that utilizes the BasicQhead for both evaluation and target networks.

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
  xuance.tensorflow.policies.gaussian_marl.BasicQnetwork.call(inputs)

  The forward method computes the evaluation Q-values for a given observation and agent identifiers, 
  along with other relevant information from the representation module.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: A tuple that includes the representation (outputs), argmax action (argmax_action), and evaluation Q-values (evalQ). These values can be useful for further processing during reinforcement learning training or evaluation.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.gaussian_marl.BasicQnetwork.target_Q(inputs)

  The target_Q method computes the target Q-values for a given observation and agent identifiers using the target Q-head. 
  This method is typically used during the training process for updating the Q-network parameters based on the temporal difference error between the evaluation Q-values and the target Q-values.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: The target Q-values.
  :rtype: torch.Tensor

.. py:function::
  xuance.tensorflow.policies.gaussian_marl.BasicQnetwork.copy_target()

  Copies the parameters from the evaluation representation, target representation, evaluation Q-head, and target Q-head.


.. py:class::
  xuance.tensorflow.policies.gaussian_marl.ActorNet(state_dim, n_agents, action_dim, hidden_sizes, normalize, initialize, activation, device)

  Represents the actor network, responsible for generating actions based on the given state and agent information. 
  It uses a Diagonal Gaussian distribution for the actions.

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
  xuance.tensorflow.policies.gaussian_marl.ActorNet.call(x)

  Passes the input tensor x through the sequential model (self.mu) to obtain the mean of the Gaussian distribution.
  Sets the parameters of the diagonal Gaussian distribution (self.dist) using the mean and the exponential of the log standard deviation.
  Returns the mean and standard deviation of the Gaussian distribution.


  :param x: The input tensor.
  :type x: tf.Tensor
  :return: The mean and standard deviation of the Gaussian distribution.

.. py:class::
  xuance.tensorflow.policies.gaussian_marl.CriticNet(state_dim, n_agents, hidden_sizes, normalize, initialize, activation, device)

  Represents the critic network, which evaluates the state-action pairs.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.tensorflow.policies.gaussian_marl.CriticNet.call(x)

  Passes the input tensor x through the sequential model (self.model) to obtain the output, 
  which represents the Q-values for the given state-action pairs.
  Returns the Q-values

  :param x: The input tensor.
  :type x: tf.Tensor
  :return: The Q-values.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.policies.gaussian_marl.MAAC_Policy(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  A multi-agent actor-critic policy with Gaussian policies. 
  It combines an actor network and a critic network and optionally uses a mixer to calculate the total team values.

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
  xuance.tensorflow.policies.gaussian_marl.MAAC_Policy.call(inputs, *rnn_hidden)

  Depending on whether the policy uses RNN, the observation is passed through the representation network, and the hidden states are updated.
  The actor network is then applied to the concatenated input of agent states and IDs to obtain the probability distribution over actions (self.pi_dist).
  Returns the updated hidden states (if RNN is used) and the probability distribution.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :param rnn_hidden: The last final hidden states of the sequence.
  :type rnn_hidden: tf.Tensor
  :return: A tuple that includes the updated hidden states (if RNN is used) and the probability distribution.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.gaussian_marl.MAAC_Policy.get_values(critic_in, agent_ids, *rnn_hidden)

  Computes the critic values based on the input states, agent IDs, and optional RNN hidden states.

  :param critic_in: The input variables of critic networks.
  :type critic_in: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :param rnn_hidden: The last final hidden states of the sequence.
  :type rnn_hidden: tf.Tensor
  :return: The critic values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.gaussian_marl.MAAC_Policy.value_tot(values_n, global_state)

  Computes the total team value, incorporating a mixer if provided.

  :param values_n: The joint values of n agents.
  :type values_n: tf.Tensor
  :param global_state: The global states of the environments.
  :type global_state: tf.Tensor
  :return: The total team value.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.gaussian_marl.MAAC_Policy.trainable_param()

  Get trainbale parameters of the model.


.. py:class::
  xuance.tensorflow.policies.gaussian_marl.Basic_ISAC_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  A basic policy architecture for the Independent Soft Actor-Critic (ISAC) algorithm, with independent actors and centralized critics. 
  It includes actor and critic networks, as well as target networks for stability during training. 
  The soft_update method is used to update the target networks gradually.

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
  xuance.tensorflow.policies.gaussian_marl.Basic_ISAC_policy.call(inputs)

  Passes the observation through the agent representation network to obtain relevant features (outputs).
  Concatenates the state features with agent identifiers and passes them through the actor network to obtain actions (act).
  Returns the representation outputs and actions.


  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: A tuple that includes the representation outputs and actions.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.gaussian_marl.Basic_ISAC_policy.critic(observation, actions, agent_ids)

  Computes critic values for given observations, actions, and agent identifiers.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param actions: The actions input.
  :type actions: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :return: The evaluated critic values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.gaussian_marl.Basic_ISAC_policy.target_critic(observation, actions, agent_ids)

  Computes critic values for target critic network given observations, actions, and agent identifiers.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param actions: The actions input.
  :type actions: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :return: The target critic values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.gaussian_marl.Basic_ISAC_policy.target_actor(observation, agent_ids)

  Obtains the output of the target actor network.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :return: The output of the target actor network.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.gaussian_marl.Basic_ISAC_policy.soft_update(tau)

  Performs a soft update of the target networks using a parameter tau.
  Updates the target actor and target critic networks by blending their parameters with the corresponding parameters of the online networks.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float


.. py:class::
  xuance.tensorflow.policies.gaussian_marl.MASAC_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  An extension of Basic_ISAC_policy for multi-agent environments. 
  It is an implementation of Multi-Agent Soft Actor-Critic (MASAC) algorithm.
  It includes modifications to the critic network to handle multiple agents.

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
  xuance.tensorflow.policies.gaussian_marl.MASAC_policy.critic(observation, actions, agent_ids)

  Computes critic values for given observations, actions, and agent identifiers. 
  Reshapes the state and actions for multiple agents.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param actions: The actions input.
  :type actions: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :return: The evaluated critic values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.gaussian_marl.MASAC_policy.target_critic(observation, actions, agent_ids)

  Computes critic values for the target critic network given observations, actions, and agent identifiers. 
  Reshapes the state and actions for multiple agents.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :param actions: The actions input.
  :type actions: tf.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: tf.Tensor
  :return: The target critic values.
  :rtype: tf.Tensor

.. raw:: html

    <br><hr>


MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.policies.gaussian_marl.BasicQhead(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation)

  A basic module representing the Q-value head, commonly used in Q-networks. 
  It takes state, action, and agent information as input and produces Q-values as output.

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
  xuance.mindspore.policies.gaussian_marl.BasicQhead.construct(x)

  Passes the input tensor x through the sequential model (self.model) and returns the output.

  :param x: The input tensor.
  :type x: ms.Tensor
  :return: The estimated Q-values.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.gaussian_marl.BasicQnetwork(action_space, n_agents, representation, hidden_sizes, normalize, initialize, activation)

  A basic Q-network that utilizes the BasicQhead for both evaluation and target networks.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: The representation module.
  :type representation: nn.Cell
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell

.. py:function::
  xuance.mindspore.policies.gaussian_marl.BasicQnetwork.construct(observation, agent_ids)

  The forward method computes the evaluation Q-values for a given observation and agent identifiers, 
  along with other relevant information from the representation module.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: A tuple that includes the representation (outputs), argmax action (argmax_action), and evaluation Q-values (evalQ). These values can be useful for further processing during reinforcement learning training or evaluation.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.gaussian_marl.BasicQnetwork.target_Q(observation, agent_ids)

  The target_Q method computes the target Q-values for a given observation and agent identifiers using the target Q-head. 
  This method is typically used during the training process for updating the Q-network parameters based on the temporal difference error between the evaluation Q-values and the target Q-values.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The target Q-values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.gaussian_marl.BasicQnetwork.copy_target()

  Copies the parameters from the evaluation representation, target representation, evaluation Q-head, and target Q-head.


.. py:class::
  xuance.mindspore.policies.gaussian_marl.ActorNet(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation)

  Represents the actor network, responsible for generating actions based on the given state and agent information. 
  It uses a Diagonal Gaussian distribution for the actions.

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
  xuance.mindspore.policies.gaussian_marl.ActorNet.construct(x)

  Passes the input tensor x through the sequential model (self.mu) to obtain the mean of the Gaussian distribution.
  Sets the parameters of the diagonal Gaussian distribution (self.dist) using the mean and the exponential of the log standard deviation.
  Returns the mean values of the Gaussian distribution.

  :param x: The input tensor.
  :type x: ms.Tensor
  :return: the mean values of the Gaussian distribution.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.gaussian_marl.CriticNet(state_dim, n_agents, hidden_sizes, normalize, initialize, activation)

  Represents the critic network, which evaluates the state-action pairs.

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
  xuance.mindspore.policies.gaussian_marl.CriticNet.construct(x)

  Passes the input tensor x through the sequential model (self.model) to obtain the output, 
  which represents the Q-values for the given state-action pairs.
  Returns the Q-values

  :param x: The input tensor.
  :type x: ms.Tensor
  :return: The Q-values.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.gaussian_marl.MAAC_Policy(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, kwargs)

  A multi-agent actor-critic policy with Gaussian policies. 
  It combines an actor network and a critic network and optionally uses a mixer to calculate the total team values.

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
  :param kwargs: The other arguments.
  :type kwargs: dict

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MAAC_Policy.construct(observation, agent_ids, *rnn_hidden, **kwargs)

  Depending on whether the policy uses RNN, the observation is passed through the representation network, and the hidden states are updated.
  The actor network is then applied to the concatenated input of agent states and IDs to obtain the probability distribution over actions (self.pi_dist).
  Returns the updated hidden states (if RNN is used) and the probability distribution.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :param kwargs: The other arguments.
  :return: A tuple that includes the updated hidden states (if RNN is used) and the probability distribution.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MAAC_Policy.get_values(observation, agent_ids, *rnn_hidden, **kwargs)

  Computes the critic values based on the input states, agent IDs, and optional RNN hidden states.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :param rnn_hidden: The final hidden state of the sequence.
  :param kwargs: The other arguments.
  :type kwargs: dict
  :return: The critic values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MAAC_Policy.value_tot(values_n, global_state)

  Computes the total team value, incorporating a mixer if provided.

  :param values_n: The joint values of n agents.
  :type values_n: ms.Tensor
  :param global_state: The global states of the environments.
  :type global_state: ms.Tensor
  :return: The total team value.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.gaussian_marl.Basic_ISAC_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  A basic policy architecture for the Independent Soft Actor-Critic (ISAC) algorithm, with independent actors and centralized critics. 
  It includes actor and critic networks, as well as target networks for stability during training. 
  The soft_update method is used to update the target networks gradually.

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
  xuance.mindspore.policies.gaussian_marl.Basic_ISAC_policy.construct(observation, agent_ids)

  Passes the observation through the agent representation network to obtain relevant features (outputs).
  Concatenates the state features with agent identifiers and passes them through the actor network to obtain actions (act).
  Returns the representation outputs and actions.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: A tuple that includes the representation outputs and actions.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.gaussian_marl.Basic_ISAC_policy.critic(observation, actions, agent_ids)

  Computes critic values for given observations, actions, and agent identifiers.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param actions: The actions input.
  :type actions: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The evaluated critic values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.gaussian_marl.Basic_ISAC_policy.critic_for_train(observation, actions, agent_ids)

  Computes critic values for given observations, actions, and agent identifiers.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param actions: The actions input.
  :type actions: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The evaluated critic values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.gaussian_marl.Basic_ISAC_policy.target_critic(observation, actions, agent_ids)

  Computes critic values for target critic network given observations, actions, and agent identifiers.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param actions: The actions input.
  :type actions: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The target critic values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.gaussian_marl.Basic_ISAC_policy.target_actor(observation, agent_ids)

  Obtains the output of the target actor network.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The output of the target actor network.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.gaussian_marl.Basic_ISAC_policy.soft_update(tau)

  Performs a soft update of the target networks using a parameter tau.
  Updates the target actor and target critic networks by blending their parameters with the corresponding parameters of the online networks.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float


.. py:class::
  xuance.mindspore.policies.gaussian_marl.MASAC_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  An extension of Basic_ISAC_policy for multi-agent environments. 
  It is an implementation of Multi-Agent Soft Actor-Critic (MASAC) algorithm.
  It includes modifications to the critic network to handle multiple agents.

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
  xuance.mindspore.policies.gaussian_marl.MASAC_policy.construct(observation, agent_ids)

  Computes critic values for given observations, actions, and agent identifiers. 
  Reshapes the state and actions for multiple agents.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The evaluated critic values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MASAC_policy.critic(observation, actions, agent_ids)

  Computes critic values for given observations, actions, and agent identifiers. 
  Reshapes the state and actions for multiple agents.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param actions: The actions input.
  :type actions: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The evaluated critic values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MASAC_policy.critic_for_train(observation, actions, agent_ids)

  Computes critic values for given observations, actions, and agent identifiers. 
  Reshapes the state and actions for multiple agents.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param actions: The actions input.
  :type actions: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The evaluated critic values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MASAC_policy.target_critic(observation, actions, agent_ids)

  Computes critic values for the target critic network given observations, actions, and agent identifiers. 
  Reshapes the state and actions for multiple agents.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param actions: The actions input.
  :type actions: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The target critic values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MASAC_policy.target_actor(observation, agent_ids)

  Obtains the output of the target actor network.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :param agent_ids: The IDs variables for agents.
  :type agent_ids: ms.Tensor
  :return: The output of the target actor network.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MASAC_policy.soft_update(tau)

  Performs a soft update of the target networks using a parameter tau.
  Updates the target actor and target critic networks by blending their parameters with the corresponding parameters of the online networks.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        import torch.distributions
        from torch.distributions.multivariate_normal import MultivariateNormal

        from xuance.torch.policies import *
        from xuance.torch.utils import *


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
                         device: Optional[Union[str, int, torch.device]] = None):
                super(BasicQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes

                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                             hidden_size, normalize, initialize, activation, device)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)

            def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
                outputs = self.representation(observation)
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
                evalQ = self.eval_Qhead(q_inputs)
                argmax_action = evalQ.argmax(dim=-1, keepdim=False)
                return outputs, argmax_action, evalQ

            def target_Q(self, observation: torch.Tensor, agent_ids: torch.Tensor):
                outputs = self.representation(observation)
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
                return self.target_Qhead(q_inputs)

            def copy_target(self):
                for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
                    tp.data.copy_(ep)


        class ActorNet(nn.Module):
            def __init__(self,
                         state_dim: int,
                         n_agents: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(ActorNet, self).__init__()
                self.device = device
                layers = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
                    layers.extend(mlp)
                layers.append(nn.Linear(hidden_sizes[0], action_dim, device=device))
                # layers.append(nn.Sigmoid())
                self.mu = nn.Sequential(*layers)
                self.log_std = nn.Parameter(-torch.ones((action_dim,), device=device))
                self.dist = DiagGaussianDistribution(action_dim)

            def forward(self, x: torch.Tensor):
                self.dist.set_param(self.mu(x), self.log_std.exp())
                return self.dist


        class CriticNet(nn.Module):
            def __init__(self,
                         state_dim: int,
                         n_agents: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None
                         ):
                super(CriticNet, self).__init__()
                layers = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.tensor):
                return self.model(x)


        class MAAC_Policy(nn.Module):
            """
            MAAC_Policy: Multi-Agent Actor-Critic Policy with Gaussian policies
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
                self.action_dim = action_space.shape[0]
                self.n_agents = n_agents
                self.representation = representation[0]
                self.representation_critic = representation[1]
                self.representation_info_shape = self.representation.output_shapes
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.actor = ActorNet(self.representation.output_shapes['state'][0], n_agents, self.action_dim,
                                      actor_hidden_size, normalize, initialize, activation, device)
                dim_input_critic = self.representation_critic.output_shapes['state'][0]
                self.critic = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                        normalize, initialize, activation, device)
                self.mixer = mixer
                self.pi_dist = None

            def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                        *rnn_hidden: torch.Tensor, **kwargs):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                actor_input = torch.concat([outputs['state'], agent_ids], dim=-1)
                self.pi_dist = self.actor(actor_input)
                return rnn_hidden, self.pi_dist

            def get_values(self, critic_in: torch.Tensor, agent_ids: torch.Tensor,
                           *rnn_hidden: torch.Tensor, **kwargs):
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


        class Basic_ISAC_policy(nn.Module):
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
                super(Basic_ISAC_policy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.n_agents = n_agents
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes

                self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                          actor_hidden_size, normalize, initialize, activation, device)
                dim_input_critic = representation.output_shapes['state'][0] + self.action_dim
                self.critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                            normalize, initialize, activation, device)
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


        class MASAC_policy(Basic_ISAC_policy):
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
                super(MASAC_policy, self).__init__(action_space, n_agents, representation,
                                                   actor_hidden_size, critic_hidden_size,
                                                   normalize, initialize, activation, device)
                dim_input_critic = (representation.output_shapes['state'][0] + self.action_dim) * self.n_agents
                self.critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                            normalize, initialize, activation, device)
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




  .. group-tab:: TensorFlow

    .. code-block:: python

        from xuance.tensorflow.policies import *
        from xuance.tensorflow.utils import *
        from xuance.tensorflow.representations import Basic_Identical
        import tensorflow_probability as tfp

        tfd = tfp.distributions


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
                        device: str = "cpu:0"):
                super(BasicQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes

                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                            hidden_size, normalize, initializer, activation, device)
                self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                              hidden_size, normalize, initializer, activation, device)
                self.copy_target()

            def call(self, inputs: Union[np.ndarray, dict], **kwargs):
                observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
                IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
                outputs = self.representation(observations)
                q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
                evalQ = tf.reshape(self.eval_Qhead(q_inputs), [-1, self.n_agents, self.action_dim])
                argmax_action = tf.argmax(evalQ, axis=-1)
                return outputs, argmax_action, evalQ

            def target_Q(self, inputs: Union[np.ndarray, dict]):
                shape_obs = inputs["obs"].shape
                shape_ids = inputs["ids"].shape
                observations = tf.reshape(inputs['obs'], [-1, shape_obs[-1]])
                IDs = tf.reshape(inputs['ids'], [-1, shape_ids[-1]])
                outputs = self.representation(observations)
                q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
                return tf.reshape(self.target_Qhead(q_inputs), shape_obs[0:-1] + (self.action_dim,))

            def copy_target(self):
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


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
                self.device = device
                layers = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
                    layers.extend(mlp)
                # layers.extend(mlp_block(input_shape[0], action_dim, None, nn.ReLU, initialize, device)[0])
                # self.mu = tk.Sequential(*layers)
                # self.logstd = tk.Sequential(*layers)
                self.outputs = tk.Sequential(layers)
                self.out_mu = tk.layers.Dense(units=action_dim, input_shape=(hidden_sizes[0],))
                self.out_std = tk.layers.Dense(units=action_dim, input_shape=(hidden_sizes[0],))

            def call(self, x: tf.Tensor, **kwargs):
                output = self.outputs(x)
                mu = tf.sigmoid(self.out_mu(output))
                std = tf.clip_by_value(self.out_std(output), -20, 1)
                std = tf.exp(std)
                return mu, std


        class CriticNet(tk.Model):
            def __init__(self,
                        state_dim: int,
                        n_agents: int,
                        hidden_sizes: Sequence[int],
                        normalize: Optional[tk.layers.Layer] = None,
                        initializer: Optional[tk.initializers.Initializer] = None,
                        activation: Optional[tk.layers.Layer] = None,
                        device: str = "cpu:0"
                        ):
                super(CriticNet, self).__init__()
                layers = []
                input_shape = (state_dim + n_agents,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initializer, device)[0])
                self.model = tk.Sequential(layers)

            def call(self, x: tf.Tensor, **kwargs):
                return self.model(x)


        class MAAC_Policy(tk.Model):
            """
            MAAC_Policy: Multi-Agent Actor-Critic Policy with Gaussian policies
            """

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
                self.action_dim = action_space.shape[0]
                self.n_agents = n_agents
                self.representation = representation[0]
                self.representation_critic = representation[1]
                self.representation_info_shape = self.representation.output_shapes
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.actor = ActorNet(self.representation.output_shapes['state'][0], n_agents, self.action_dim,
                                      actor_hidden_size, normalize, initialize, activation, device)
                dim_input_critic = self.representation_critic.output_shapes['state'][0]
                self.critic = CriticNet(dim_input_critic, n_agents,  critic_hidden_size,
                                        normalize, initialize, activation, device)
                self.mixer = mixer
                self.identical_rep = True if isinstance(self.representation, Basic_Identical) else False
                self.pi_dist = None

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
                mu, std = self.actor(actor_input)
                mu = tf.reshape(mu, [-1, self.n_agents, self.action_dim])
                std = tf.reshape(std, [-1, self.n_agents, self.action_dim])
                cov_mat = tf.linalg.diag(std)
                dist = tfd.MultivariateNormalTriL(loc=mu, scale_tril=cov_mat)
                return rnn_hidden, dist

            def get_values(self, critic_in: tf.Tensor, agent_ids: tf.Tensor, *rnn_hidden: tf.Tensor, **kwargs):
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
                    global_state = torch.as_tensor(global_state).to(self.device)
                return values_n if self.mixer is None else self.mixer(values_n, global_state)

            def trainable_param(self):
                params = self.actor.trainable_variables + self.critic.trainable_variables
                if self.mixer is not None:
                    params += self.mixer.trainable_variables
                if self.identical_rep:
                    return params
                else:
                    return params + self.representation.trainable_variables


        class Basic_ISAC_policy(tk.Model):
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
                super(Basic_ISAC_policy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.n_agents = n_agents
                self.representation = representation
                self.obs_dim = self.representation.input_shapes[0]
                self.representation_info_shape = self.representation.output_shapes

                self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                          actor_hidden_size, normalize, initializer, activation, device)
                dim_input_critic = representation.output_shapes['state'][0] + self.action_dim
                self.critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                            normalize, initializer, activation, device)
                self.target_actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                                actor_hidden_size, normalize, initializer, activation, device)
                self.target_critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                                  normalize, initializer, activation, device)
                if isinstance(self.representation, Basic_Identical):
                    self.parameters_actor = self.actor_net.trainable_variables
                else:
                    self.parameters_actor = self.representation.trainable_variables + self.actor_net.trainable_variables
                self.parameters_critic = self.critic_net.trainable_variables
                self.soft_update(tau=1.0)

            def call(self, inputs: Union[np.ndarray, dict], **kwargs):
                observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
                IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
                outputs = self.representation(observations)
                actor_in = tf.concat([outputs['state'], IDs], axis=-1)
                mu, std = self.actor_net(actor_in)
                mu = tf.reshape(mu, [-1, self.n_agents, self.action_dim])
                std = tf.reshape(std, [-1, self.n_agents, self.action_dim])
                cov_mat = tf.linalg.diag(std)
                dist = tfd.MultivariateNormalTriL(loc=mu, scale_tril=cov_mat)
                return outputs, dist

            def critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
                outputs = self.representation(observation)
                critic_in = tf.concat([outputs['state'], actions, agent_ids], axis=-1)
                return self.critic_net(critic_in)

            def target_critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
                outputs = self.representation(observation)
                critic_in = tf.concat([outputs['state'], actions, agent_ids], axis=-1)
                return self.target_critic_net(critic_in)

            def target_actor(self, observation: tf.Tensor, agent_ids: tf.Tensor):
                outputs = self.representation(observation)
                actor_in = tf.concat([outputs['state'], agent_ids], axis=-1)
                mu, std = self.target_actor_net(actor_in)
                mu = tf.reshape(mu, [-1, self.n_agents, self.action_dim])
                std = tf.reshape(std, [-1, self.n_agents, self.action_dim])
                cov_mat = tf.linalg.diag(std)
                dist = tfd.MultivariateNormalTriL(loc=mu, scale_tril=cov_mat)
                return dist

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor_net.variables, self.target_actor_net.variables):
                    tp.assign((1 - tau) * tp + tau * ep)
                for ep, tp in zip(self.critic_net.variables, self.target_critic_net.variables):
                    tp.assign((1 - tau) * tp + tau * ep)


        class MASAC_policy(Basic_ISAC_policy):
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
                super(MASAC_policy, self).__init__(action_space, n_agents, representation,
                                                  actor_hidden_size, critic_hidden_size,
                                                  normalize, initializer, activation, device)
                dim_input_critic = (representation.output_shapes['state'][0] + self.action_dim) * self.n_agents
                self.critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                            normalize, initializer, activation, device)
                self.target_critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                                  normalize, initializer, activation, device)
                self.parameters_critic = self.critic_net.trainable_variables
                self.soft_update(tau=1.0)

            def critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
                bs = observation.shape[0]
                outputs_n = self.representation(observation)['state']
                outputs_n = tf.tile(tf.reshape(outputs_n, [bs, 1, -1]), (1, self.n_agents, 1))
                actions_n = tf.tile(tf.reshape(actions, [bs, 1, -1]), (1, self.n_agents, 1))
                critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
                return self.critic_net(critic_in)

            def target_critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
                bs = observation.shape[0]
                outputs_n = self.representation(observation)['state']
                outputs_n = tf.tile(tf.reshape(outputs_n, [bs, 1, -1]), (1, self.n_agents, 1))
                actions_n = tf.tile(tf.reshape(actions, [bs, 1, -1]), (1, self.n_agents, 1))
                critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
                return self.target_critic_net(critic_in)



  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.policies import *
        from xuance.mindspore.utils import *
        from xuance.mindspore.representations import Basic_Identical
        from mindspore.nn.probability.distribution import Normal
        import copy


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
                        activation: Optional[ModuleType] = None):
                super(BasicQnetwork, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes

                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                            hidden_size, normalize, initialize, activation)
                self.target_Qhead = copy.deepcopy(self.eval_Qhead)
                self._concat = ms.ops.Concat(axis=-1)

            def construct(self, observation: ms.tensor, agent_ids: ms.tensor):
                outputs = self.representation(observation)
                q_inputs = self._concat([outputs['state'], agent_ids])
                evalQ = self.eval_Qhead(q_inputs)
                argmax_action = evalQ.argmax(dim=-1, keepdim=False)
                return outputs, argmax_action, evalQ

            def target_Q(self, observation: ms.tensor, agent_ids: ms.tensor):
                outputs = self.representation(observation)
                q_inputs = self._concat([outputs['state'], agent_ids])
                return self.target_Qhead(q_inputs)

            def copy_target(self):
                for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
                    tp.assign_value(ep)


        class ActorNet(nn.Cell):
            class Sample(nn.Cell):
                def __init__(self, log_std):
                    super(ActorNet.Sample, self).__init__()
                    self._dist = Normal(dtype=ms.float32)
                    self.logstd = log_std
                    self._exp = ms.ops.Exp()

                def construct(self, mean: ms.tensor):
                    return self._dist.sample(mean=mean, sd=self._exp(self.logstd))

            class LogProb(nn.Cell):
                def __init__(self, log_std):
                    super(ActorNet.LogProb, self).__init__()
                    self._dist = Normal(dtype=ms.float32)
                    self.logstd = log_std
                    self._exp = ms.ops.Exp()
                    self._sum = ms.ops.ReduceSum(keep_dims=False)

                def construct(self, value: ms.tensor, probs: ms.tensor):
                    return self._sum(self._dist.log_prob(value, probs, self._exp(self.logstd)), -1)

            class Entropy(nn.Cell):
                def __init__(self, log_std):
                    super(ActorNet.Entropy, self).__init__()
                    self._dist = Normal(dtype=ms.float32)
                    self.logstd = log_std
                    self._exp = ms.ops.Exp()
                    self._sum = ms.ops.ReduceSum(keep_dims=False)

                def construct(self, probs: ms.tensor):
                    return self._sum(self._dist.entropy(probs, self._exp(self.logstd)), -1)

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
                layers.extend(mlp_block(input_shape[0], action_dim, None, None, initialize)[0])
                self.mu = nn.SequentialCell(*layers)
                self._ones = ms.ops.Ones()
                self.logstd = ms.Parameter(-self._ones((action_dim,), ms.float32))
                # define the distribution methods
                self.sample = self.Sample(self.logstd)
                self.log_prob = self.LogProb(self.logstd)
                self.entropy = self.Entropy(self.logstd)

            def construct(self, x: ms.tensor):
                return self.mu(x)


        class CriticNet(nn.Cell):
            def __init__(self,
                        state_dim: int,
                        n_agents: int,
                        hidden_sizes: Sequence[int],
                        normalize: Optional[ModuleType] = None,
                        initialize: Optional[Callable[..., ms.Tensor]] = None,
                        activation: Optional[ModuleType] = None
                        ):
                super(CriticNet, self).__init__()
                layers = []
                input_shape = (state_dim + n_agents, )
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initialize)[0])
                self.model = nn.SequentialCell(*layers)

            def construct(self, x: ms.tensor):
                return self.model(x)


        class MAAC_Policy(nn.Cell):
            """
            MAAC_Policy: Multi-Agent Actor-Critic Policy with Gaussian policies
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
                        **kwargs):
                super(MAAC_Policy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.n_agents = n_agents
                self.representation = representation[0]
                self.representation_critic = representation[1]
                self.representation_info_shape = self.representation.output_shapes
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.use_rnn = True if kwargs["use_recurrent"] else False
                self.actor = ActorNet(self.representation.output_shapes['state'][0], n_agents, self.action_dim,
                                      actor_hidden_size, normalize, initialize, activation)
                dim_input_critic = self.representation_critic.output_shapes['state'][0]
                self.critic = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                        normalize, initialize, activation)
                self.mixer = mixer
                self._concat = ms.ops.Concat(axis=-1)

            def construct(self, observation: ms.tensor, agent_ids: ms.tensor,
                          *rnn_hidden: ms.tensor, **kwargs):
                if self.use_rnn:
                    outputs = self.representation(observation, *rnn_hidden)
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.representation(observation)
                    rnn_hidden = None
                actor_input = self._concat([outputs['state'], agent_ids])
                mu_a = self.actor(actor_input)
                return rnn_hidden, mu_a

            def get_values(self, critic_in: ms.tensor, agent_ids: ms.tensor, *rnn_hidden: ms.tensor, **kwargs):
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

            def value_tot(self, values_n: ms.tensor, global_state=None):
                if global_state is not None:
                    global_state = torch.as_tensor(global_state).to(self.device)
                return values_n if self.mixer is None else self.mixer(values_n, global_state)


        class Basic_ISAC_policy(nn.Cell):
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
                super(Basic_ISAC_policy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.n_agents = n_agents
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes

                self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                          actor_hidden_size, normalize, initialize, activation)
                dim_input_critic = representation.output_shapes['state'][0] + self.action_dim
                self.critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size, normalize, initialize, activation)
                self.target_actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                                actor_hidden_size, normalize, initialize, activation)
                self.target_critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                                  normalize, initialize, activation)
                self.parameters_actor = list(self.representation.trainable_params()) + list(self.actor_net.trainable_params())
                self.parameters_critic = self.critic_net.trainable_params()
                self._concat = ms.ops.Concat(axis=-1)
                self.soft_update(tau=1.0)

            def construct(self, observation: ms.tensor, agent_ids: ms.tensor):
                outputs = self.representation(observation)
                actor_in = self._concat([outputs['state'], agent_ids])
                mu_a = self.actor_net(actor_in)
                return outputs, mu_a

            def critic(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
                outputs = self.representation(observation)
                critic_in = self._concat([outputs['state'], actions, agent_ids])
                return self.critic_net(critic_in)

            def critic_for_train(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
                outputs = self.representation(observation)
                critic_in = self._concat([outputs['state'], actions, agent_ids])
                return self.critic_net(critic_in)

            def target_critic(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
                outputs = self.representation(observation)
                critic_in = self._concat([outputs['state'], actions, agent_ids])
                return self.target_critic_net(critic_in)

            def target_actor(self, observation: ms.tensor, agent_ids: ms.tensor):
                outputs = self.representation(observation)
                actor_in = self._concat([outputs['state'], agent_ids])
                mu_a = self.target_actor_net(actor_in)
                return mu_a

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
                for ep, tp in zip(self.critic_net.trainable_params(), self.target_critic_net.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


        class MASAC_policy(nn.Cell):
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
                super(MASAC_policy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.n_agents = n_agents
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes

                self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                          actor_hidden_size, normalize, initialize, activation)
                dim_input_critic = (representation.output_shapes['state'][0] + self.action_dim) * self.n_agents
                self.critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size, normalize, initialize, activation)
                self.target_actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                                actor_hidden_size, normalize, initialize, activation)
                self.target_critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                                  normalize, initialize, activation)
                self.parameters_actor = list(self.representation.trainable_params()) + list(self.actor_net.trainable_params())
                self.parameters_critic = self.critic_net.trainable_params()
                self._concat = ms.ops.Concat(axis=-1)
                self.soft_update(tau=1.0)
                self.broadcast_to = ms.ops.BroadcastTo((-1, self.n_agents, -1))
                self.broadcast_to_act = ms.ops.BroadcastTo((-1, self.n_agents, -1))

            def construct(self, observation: ms.tensor, agent_ids: ms.tensor):
                outputs = self.representation(observation)
                actor_in = self._concat([outputs['state'], agent_ids])
                mu_a = self.actor_net(actor_in)
                return outputs, mu_a

            def critic(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
                bs = observation.shape[0]
                outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
                actions_n = self.broadcast_to_act(actions.view(bs, 1, -1))
                critic_in = self._concat([outputs_n, actions_n, agent_ids])
                return self.critic_net(critic_in)

            def critic_for_train(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
                bs = observation.shape[0]
                outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
                actions_n = self.broadcast_to_act(actions.view(bs, 1, -1))
                critic_in = self._concat([outputs_n, actions_n, agent_ids])
                return self.critic_net(critic_in)

            def target_critic(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
                bs = observation.shape[0]
                outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
                actions_n = self.broadcast_to_act(actions.view(bs, 1, -1))
                critic_in = self._concat([outputs_n, actions_n, agent_ids])
                return self.target_critic_net(critic_in)

            def target_actor(self, observation: ms.tensor, agent_ids: ms.tensor):
                outputs = self.representation(observation)
                actor_in = self._concat([outputs['state'], agent_ids])
                mu_a = self.target_actor_net(actor_in)
                return mu_a

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
                for ep, tp in zip(self.critic_net.trainable_params(), self.target_critic_net.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


