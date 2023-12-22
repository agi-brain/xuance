Categorical
======================================

The categorical policies are designed for environments with discrete actions space. 
They calculate action distributions according to the outputs of actor networks, 
and generate actions by random sampling from these distributions. 

In this module, we provide two basic categorical policies ( ``ActorPolicy`` and ``ActorCriticPolicy``)
and numerous specially-made policies (e.g., ``PPGActorCritic``, ``SACDISPolicy``, etc.).
You can also customize the other categorical policies for single agent DRL here.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.policies.categorical.ActorNet(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

  A neural network module for an actor in DRL, used as an important part of a policy. 
  This actor network outputs categorical actions distributions based on the given state.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.torch.policies.categorical.ActorNet.forward(x)

  A feed forward method that is used to calculate the model's output based on the input state x,
  and set the parameters of the categorical distribution based on the output of the actor model.
  Finally, this method returns a distribution.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: A categorical distribution that can be used to sample actions.

.. py:class::
  xuance.torch.policies.categorical.CriticNet(state_dim, hidden_sizes, normalize, initialize, activation, device)
  
  This is a neural network module, CriticNet, typically used for approximating the value function of a state. 

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.torch.policies.categorical.CriticNet.forward(x)

  A feed forward method that takes a tensor x as input and passes it through the critic network, 
  returning the output (value estimate).

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: The evaluated values of the input x.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.categorical.ActorCriticPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  A module for an actor-critic policy in DRL. 
  This type of policy is commonly used in actor-critic DRL algorithms, 
  where the actor is responsible for selecting actions, and the critic evaluates the value of the current state.

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  xuance.torch.policies.categorical.ActorCriticPolicy.forward(observation)

  A feed forward method that takes an observation (input state) and performs a forward pass through the representation, actor, and critic networks. 
  It returns the outputs of the representation (hidden states), actor (action distributions), and critic (evaluated values).

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :return: A tuple that includes the outputs of the representation (hidden states), actor (action distributions), and critic (evaluated values).
  :rtype: tuple

.. py:class::
  xuance.torch.policies.categorical.ActorPolicy(action_space, representation, actor_hidden_size, normalize, initialize, activation, device)

  This type of policy is commonly used in actor-only reinforcement learning algorithms, 
  where the actor is responsible for selecting actions based on the current state.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.categorical.ActorPolicy.forward(observation)

  A feed forward method that takes an observation (input state) and performs a forward pass through the representation and actor networks. 
  It returns the outputs of the representation (hidden states) and actor (categorical action distributions).

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :return: A tuple that includes the outputs of the representation (hidden states) and actor (categorical action distributions).
  :rtype: tuple

.. py:class::
  xuance.torch.policies.categorical.PPGActorCritic(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  An implementation of an actor-critic model for phasic policy gradient methods in reinforcement learning.

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  xuance.torch.policies.categorical.PPGActorCritic.forward(observation)

  A feed forward method that takes an observation (input state) and performs a forward pass through the representation networks, actor, critic, and auxiliary critic networks. 
  It returns the outputs of these components.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :return: A tuple that includes the outputs of these representation networks, actor, critic, and auxiliary critic networks.
  :rtype: tuple

.. py:class::
  xuance.torch.policies.categorical.CriticNet_SACDIS(state_dim, action_dim, hidden_sizes, initialize, activation, device)

  An implementation of a critic network for the Soft Actor-Critic (SAC) with discrete action spaces.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: torch.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.categorical.CriticNet_SACDIS.forward(x)

  A feed forward method that defines the forward pass through the critic network, 
  taking the input tensor x and passing it through the critic model.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: The evaluated critic values of input x.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.categorical.ActorNet_SACDIS(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

  An implementation of a actor network for the Soft Actor-Critic (SAC) with discrete action spaces. 
  It takes the state as input and outputs a probability distribution over discrete actions using a softmax activation.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.torch.policies.categorical.ActorNet_SACDIS.forward(x)

  A feed forward method that takes the tensor x as input and passes it through the actor model.
  It returns a categorical distribution over discrete actions.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: A probability distribution over discrete actions using a softmax activation.

.. py:class::
  xuance.torch.policies.categorical.SACDISPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  This class defines a policy for the soft actor-critic with discrete action spaces.

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  xuance.torch.policies.categorical.SACDISPolicy.forward(observation)

  A feed forward method that computes the forward pass of the policy network given an observation. 
  It returns the representation of the observation, action probabilities, and the action distribution.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :return: A tuple that includes the outputs of the representation, action probabilities, and the action distribution.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.categorical.SACDISPolicy.Qtarget(observation)

  Calculate the Q-value with the target Q network.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :return: The target Q values.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.categorical.SACDISPolicy.Qaction(observation)

  Calculate the Q value for the original Q network.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :return: The evaluate Q values.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.categorical.SACDISPolicy.Qpolicy(observation)

  Calculate the action probabilities, log of action probabilities, and the Q-values of the policy network.

  :param observation: The original observation variables.
  :type observation: torch.Tensor
  :return: A tuple that includes the action probabilities, log of action probabilities, and the Q-values of the policy network.
  :rtype: tuple

.. py:function::
  xuance.torch.policies.categorical.SACDISPolicy.soft_update(tau)

  Performs a soft update of the target networks using a specified interpolation parameter (tau).

  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.policies.categorical.ActorNet(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

  A neural network module for an actor in DRL, used as an important part of a policy. 
  This actor network outputs the logits of the categorical actions distributions based on the given state.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.tensorflow.policies.categorical.ActorNet.call(x)

  A feed forward method that is used to calculate the model's output based on the input state x, 
  and set the parameters of the categorical distribution based on the output of the actor model. 
  Finally, this method returns the logits of the distribution.

  :param x: The input tensor.
  :type x: tf.Tensor
  :return: The logits of the distribution.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.policies.categorical.CriticNet(state_dim, hidden_sizes, normalize, initialize, activation, device)

  This is a neural network module, CriticNet, typically used for approximating the value function of a state.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.tensorflow.policies.categorical.CriticNet.call(x)

  A feed forward method that takes a tensor x as input and passes it through the critic network, returning the output (value estimate).

  :param x: The input tensor.
  :type x: tf.Tensor
  :return: The evaluated values of the input x.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.policies.categorical.ActorCriticPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  A module for an actor-critic policy in DRL. 
  This type of policy is commonly used in actor-critic DRL algorithms, 
  where the actor is responsible for selecting actions, and the critic evaluates the value of the current state.

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  xuance.tensorflow.policies.categorical.ActorCriticPolicy.call(observation)

  A feed forward method that takes an observation (input state) and performs a forward pass through the representation, actor, and critic networks. 
  It returns the outputs of the representation (hidden states), actor (logits of the action distributions), and critic (evaluated values).

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :return: A tuple that includes the outputs of the representation (hidden states), actor (logits of the action distributions), and critic (evaluated values).
  :rtype: tuple

.. py:class::
  xuance.tensorflow.policies.categorical.ActorPolicy(action_space, representation, actor_hidden_size, normalize, initialize, activation, device)

  This type of policy is commonly used in actor-only reinforcement learning algorithms, where the actor is responsible for selecting actions based on the current state.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: tk.Model
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.categorical.ActorPolicy.call(observation)

  A feed forward method that takes an observation (input state) and performs a forward pass through the representation and actor networks. 
  It returns the outputs of the representation (hidden states) and actor (logits of the categorical action distributions).

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :return: A tuple that includes the outputs of the representation (hidden states) and actor (logits of the categorical action distributions).
  :rtype: tuple

.. py:class::
  xuance.tensorflow.policies.categorical.PPGActorCritic(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  An implementation of an actor-critic model for phasic policy gradient methods in reinforcement learning.

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  xuance.tensorflow.policies.categorical.PPGActorCritic.call(observation)

  A feed forward method that takes an observation (input state) and performs a forward pass through the representation networks, actor, critic, and auxiliary critic networks. 
  It returns the outputs of these components.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :return: A tuple that includes the outputs of these representation networks, actor, critic, and auxiliary critic networks.
  :rtype: tuple

.. py:class::
  xuance.tensorflow.policies.categorical.CriticNet_SACDIS(state_dim, action_dim, hidden_sizes, initialize, activation, device)

  An implementation of a critic network for the Soft Actor-Critic (SAC) with discrete action spaces.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.categorical.CriticNet_SACDIS.call(x)

  A feed forward method that defines the forward pass through the critic network, 
  taking the input tensor x and passing it through the critic model.

  :param x: The input tensor.
  :type x: tf.Tensor
  :return: The evaluated critic values of input x.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.policies.categorical.ActorNet_SACDIS(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

  An implementation of a actor network for the Soft Actor-Critic (SAC) with discrete action spaces. 
  It takes the state as input and outputs a probability distribution over discrete actions using a softmax activation.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.tensorflow.policies.categorical.ActorNet_SACDIS.call(x)

  A feed forward method that takes the tensor x as input and passes it through the actor model. 
  It returns a categorical distribution over discrete actions.

  :param x: The input tensor.
  :type x: tf.Tensor
  :return: A tuple that includes the action probabilities and the categorical distribution over discrete actions using a softmax activation.
  :rtype: tuple

.. py:class::
  xuance.tensorflow.policies.categorical.SACDISPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  This class defines a policy for the soft actor-critic with discrete action spaces.

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  xuance.tensorflow.policies.categorical.SACDISPolicy.call(observation)

  A feed forward method that computes the forward pass of the policy network given an observation. 
  It returns the representation of the observation, action probabilities, and the action distribution.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :return: A tuple that includes the outputs of the representation, action probabilities, and the action distribution.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.categorical.SACDISPolicy.Qtarget(observation)

  Calculate the Q-value with the target Q network.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :return: The target Q values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.categorical.SACDISPolicy.Qaction(observation)

  Calculate the Q value for the original Q network.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :return: The evaluate Q values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.categorical.SACDISPolicy.Qpolicy(observation)

  Calculate the action probabilities, log of action probabilities, and the Q-values of the policy network.

  :param observation: The original observation variables.
  :type observation: tf.Tensor
  :return: A tuple that includes the action probabilities, log of action probabilities, and the Q-values of the policy network.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.policies.categorical.SACDISPolicy.soft_update(tau)

  Performs a soft update of the target networks using a specified interpolation parameter (tau).

  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.policies.categorical.ActorNet(state_dim, action_dim, hidden_sizes, normalize, initialize, activation)

  A neural network module for an actor in DRL, used as an important part of a policy. 
  This actor network outputs the logits for categorical actions distributions based on the given state.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.mindspore.policies.categorical.ActorNet.construct(x)

  A feed forward method that is used to calculate the model's output based on the input state x, 
  and set the parameters of the categorical distribution based on the output of the actor model. 
  Finally, this method returns the logits of categorical distribution.

  :param x: The input tensor.
  :type x: ms.Tensor
  :return: The logits of categorical distribution.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.categorical.CriticNet(state_dim, hidden_sizes, normalize, initialize, activation)

  This is a neural network module, CriticNet, typically used for approximating the value function of a state.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell

.. py:function::
  xuance.mindspore.policies.categorical.CriticNet.construct(x)

  A feed forward method that takes a tensor x as input and passes it through the critic network, returning the output (value estimate).

  :param x: The input tensor.
  :type x: ms.Tensor
  :return: The evaluated values of the input x.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.categorical.ActorCriticPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  A module for an actor-critic policy in DRL. 
  This type of policy is commonly used in actor-critic DRL algorithms, 
  where the actor is responsible for selecting actions, and the critic evaluates the value of the current state.

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  xuance.mindspore.policies.categorical.ActorCriticPolicy.construct(observation)

  A feed forward method that takes an observation (input state) and performs a forward pass through the representation, actor, and critic networks. 
  It returns the outputs of the representation (hidden states), actor (logits of the action distributions), and critic (evaluated values).

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :return: A tuple that includes the outputs of the representation (hidden states), actor (logits of the action distributions), and critic (evaluated values).
  :rtype: tuple

.. py:class::
  xuance.mindspore.policies.categorical.ActorPolicy(action_space, representation, actor_hidden_size, normalize, initialize, activation)

  This type of policy is commonly used in actor-only reinforcement learning algorithms, 
  where the actor is responsible for selecting actions based on the current state.

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Cell
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell

.. py:function::
  xuance.mindspore.policies.categorical.ActorPolicy.construct(observation)

  A feed forward method that takes an observation (input state) and performs a forward pass through the representation and actor networks. 
  It returns the outputs of the representation (hidden states) and actor (logits of the categorical action distributions).

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :return: A tuple that includes the outputs of the representation (hidden states) and actor (logits of the categorical action distributions).
  :rtype: tuple

.. py:class::
  xuance.mindspore.policies.categorical.PPGActorCritic(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  An implementation of an actor-critic model for phasic policy gradient methods in reinforcement learning.

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  xuance.mindspore.policies.categorical.PPGActorCritic.construct(observation)

  A feed forward method that takes an observation (input state) and performs a forward pass through the representation networks, actor, critic, and auxiliary critic networks. 
  It returns the outputs of these components.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :return: A tuple that includes the outputs of these representation networks, actor, critic, and auxiliary critic networks.
  :rtype: tuple

.. py:class::
  xuance.mindspore.policies.categorical.CriticNet_SACDIS(state_dim, action_dim, hidden_sizes, initialize, activation)

  An implementation of a critic network for the Soft Actor-Critic (SAC) with discrete action spaces.

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param action_dim: The dimension of the action input.
  :type action_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell

.. py:function::
  xuance.mindspore.policies.categorical.CriticNet_SACDIS.construct(x)

  A feed forward method that defines the forward pass through the critic network, taking the input tensor x and passing it through the critic model.

  :param x: The input tensor.
  :type x: ms.Tensor
  :return: The evaluated critic values of input x.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.categorical.SACDISPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  A feed forward method that computes the forward pass of the policy network given an observation. 
  It returns the representation of the observation, action probabilities, and the action distribution.

  :param action_space: The action space of the environment.
  :type action_space: Space
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
  xuance.mindspore.policies.categorical.SACDISPolicy.construct(observation)

  A feed forward method that computes the forward pass of the policy network given an observation. 
  It returns the representation of the observation, action probabilities, and the action distribution.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :return: A tuple that includes the outputs of the representation, action probabilities, and the action distribution.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.categorical.SACDISPolicy.action(observation)

  Get actions according to the observations.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :return: A tuple that includes the ouputs of the representation (hidden states) and the actor network (action probabilities).
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.categorical.SACDISPolicy.Qtarget(observation)

  Calculate the Q-value with the target Q network.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :return: The target Q values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.categorical.SACDISPolicy.Qaction(observation)

  Calculate the Q value for the original Q network.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :return: The evaluate Q values.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.policies.categorical.SACDISPolicy.Qpolicy(observation)

  Calculate the action probabilities, log of action probabilities, and the Q-values of the policy network.

  :param observation: The original observation variables.
  :type observation: ms.Tensor
  :return: A tuple that includes the action probabilities, log of action probabilities, and the Q-values of the policy network.
  :rtype: tuple

.. py:function::
  xuance.mindspore.policies.categorical.SACDISPolicy.soft_update(tau)

  Performs a soft update of the target networks using a specified interpolation parameter (tau).

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

        import torch.distributions

        from xuance.torch.policies import *
        from xuance.torch.utils import *
        from xuance.torch.representations import Basic_Identical


        def _init_layer(layer, gain=np.sqrt(2), bias=0.0):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, bias)
            return layer


        class ActorNet(nn.Module):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(ActorNet, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, None, initialize, device)[0])
                self.model = nn.Sequential(*layers)
                self.dist = CategoricalDistribution(action_dim)

            def forward(self, x: torch.Tensor):
                self.dist.set_param(self.model(x))
                return self.dist


        class CriticNet(nn.Module):
            def __init__(self,
                         state_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(CriticNet, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor):
                return self.model(x)[:, 0]


        class ActorCriticPolicy(nn.Module):
            def __init__(self,
                         action_space: Space,
                         representation: nn.Module,
                         actor_hidden_size: Sequence[int] = None,
                         critic_hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(ActorCriticPolicy, self).__init__()
                self.device = device
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_info_shape = representation.output_shapes
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                      normalize, initialize, activation, device)
                self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                        normalize, initialize, activation, device)

            def forward(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                a = self.actor(outputs['state'])
                v = self.critic(outputs['state'])
                return outputs, a, v


        class ActorPolicy(nn.Module):
            def __init__(self,
                         action_space: Space,
                         representation: nn.Module,
                         actor_hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(ActorPolicy, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                      normalize, initialize, activation, device)

            def forward(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                a = self.actor(outputs['state'])
                return outputs, a


        class PPGActorCritic(nn.Module):
            def __init__(self,
                         action_space: Space,
                         representation: nn.Module,
                         actor_hidden_size: Sequence[int] = None,
                         critic_hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(PPGActorCritic, self).__init__()
                self.action_dim = action_space.n
                self.actor_representation = representation
                self.critic_representation = copy.deepcopy(representation)
                self.aux_critic_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.actor_representation.output_shapes

                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                      normalize, initialize, activation, device)
                self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                        normalize, initialize, activation, device)
                self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                            normalize, initialize, activation, device)

            def forward(self, observation: Union[np.ndarray, dict]):
                policy_outputs = self.actor_representation(observation)
                critic_outputs = self.critic_representation(observation)
                aux_critic_outputs = self.aux_critic_representation(observation)
                a = self.actor(policy_outputs['state'])
                v = self.critic(critic_outputs['state'])
                aux_v = self.aux_critic(aux_critic_outputs['state'])
                return policy_outputs, a, v, aux_v


        class CriticNet_SACDIS(nn.Module):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(CriticNet_SACDIS, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, None, initialize, device)[0])
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.tensor):
                return self.model(x)


        class ActorNet_SACDIS(nn.Module):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(ActorNet_SACDIS, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
                self.output = nn.Sequential(*layers)
                self.model = nn.Softmax(dim=-1)

            def forward(self, x: torch.tensor):
                action_prob = self.model(self.output(x))
                dist = torch.distributions.Categorical(probs=action_prob)
                # action_logits = self.output(x)
                # dist = torch.distributions.Categorical(logits=action_logits)
                # action_prob = dist.probs
                return action_prob, dist


        class SACDISPolicy(nn.Module):
            def __init__(self,
                         action_space: Space,
                         representation: nn.Module,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(SACDISPolicy, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_critic = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet_SACDIS(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                             normalize, initialize, activation, device)
                self.critic = CriticNet_SACDIS(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                               initialize, activation, device)
                self.target_representation_critic = copy.deepcopy(self.representation_critic)
                self.target_critic = copy.deepcopy(self.critic)

            def forward(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                act_prob, act_distribution = self.actor(outputs['state'])
                return outputs, act_prob, act_distribution

            def Qtarget(self, observation: Union[np.ndarray, dict]):
                outputs_actor = self.representation(observation)
                outputs_critic = self.target_representation_critic(observation)
                act_prob, act_distribution = self.actor(outputs_actor['state'])
                # z = act_prob == 0.0
                # z = z.float() * 1e-8
                log_action_prob = torch.log(act_prob + 1e-5)
                return act_prob, log_action_prob, self.target_critic(outputs_critic['state'])

            def Qaction(self, observation: Union[np.ndarray, dict]):
                outputs_critic = self.representation_critic(observation)
                return outputs_critic, self.critic(outputs_critic['state'])

            def Qpolicy(self, observation: Union[np.ndarray, dict]):
                outputs_actor = self.representation(observation)
                outputs_critic = self.representation(observation)
                act_prob, act_distribution = self.actor(outputs_actor['state'])
                # z = act_prob == 0.0
                # z = z.float() * 1e-8
                log_action_prob = torch.log(act_prob + 1e-5)
                return act_prob, log_action_prob, self.critic(outputs_critic['state'])

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.representation_critic.parameters(), self.target_representation_critic.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)


  .. group-tab:: TensorFlow

    .. code-block:: python

        from xuance.tensorflow.policies import *
        from xuance.tensorflow.utils import *


        class ActorNet(tk.Model):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(ActorNet, self).__init__()
                layers = []
                input_shapes = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shapes = mlp_block(input_shapes[0], h, normalize, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shapes[0], action_dim, device=device)[0])
                self.model = tk.Sequential(layers)
                self.dist = CategoricalDistribution(action_dim)

            def call(self, x: tf.Tensor, **kwargs):
                logits = self.model(x)
                self.dist.set_param(logits)
                return logits


        class CriticNet(tk.Model):
            def __init__(self,
                         state_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(CriticNet, self).__init__()
                layers = []
                input_shapes = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shapes = mlp_block(input_shapes[0], h, normalize, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shapes[0], 1, device=device)[0])
                self.model = tk.Sequential(layers)

            def call(self, x: tf.Tensor, **kwargs):
                return self.model(x)[:, 0]


        class ActorCriticPolicy(tk.Model):
            def __init__(self,
                         action_space: Space,
                         representation: tk.Model,
                         actor_hidden_size: Sequence[int] = None,
                         critic_hidden_size: Sequence[int] = None,
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(ActorCriticPolicy, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                      normalize, initializer, activation, device)
                self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                        normalize, initializer, activation, device)

            def call(self, observations: Union[np.ndarray, dict], **kwargs):
                outputs = self.representation(observations)
                a = self.actor(outputs['state'])
                v = self.critic(outputs['state'])
                return outputs, a, v


        class ActorPolicy(tk.Model):
            def __init__(self,
                         action_space: Space,
                         representation: tk.Model,
                         actor_hidden_size: Sequence[int] = None,
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(ActorPolicy, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                      normalize, initializer, activation, device)

            def call(self, observation: Union[np.ndarray, dict], **kwargs):
                outputs = self.representation(observation)
                a = self.actor(outputs['state'])
                return outputs, a


        class PPGActorCritic(tk.Model):
            def __init__(self,
                         action_space: Space,
                         representation: tk.Model,
                         actor_hidden_size: Sequence[int] = None,
                         critic_hidden_size: Sequence[int] = None,
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                assert isinstance(action_space, Discrete)
                super(PPGActorCritic, self).__init__()
                self.action_dim = action_space.n
                self.actor_representation = representation
                self.critic_representation = copy.deepcopy(representation)
                self.aux_critic_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.actor_representation.output_shapes

                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                      normalize, initializer, activation, device)
                self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                        normalize, initializer, activation, device)
                self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                            normalize, initializer, activation, device)

            def call(self, observation: Union[np.ndarray, dict], **kwargs):
                policy_outputs = self.actor_representation(observation)
                critic_outputs = self.critic_representation(observation)
                aux_critic_outputs = self.aux_critic_representation(observation)
                a = self.actor(policy_outputs['state'])
                v = self.critic(critic_outputs['state'])
                aux_v = self.aux_critic(aux_critic_outputs['state'])
                return policy_outputs, a, v, aux_v


        class CriticNet_SACDIS(tk.Model):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(CriticNet_SACDIS, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, None, initializer, device)[0])
                self.model = tk.Sequential(layers)

            def call(self, x: tf.Tensor, **kwargs):
                return self.model(x)


        class ActorNet_SACDIS(tk.Model):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(ActorNet_SACDIS, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
                self.outputs = tk.Sequential(layers)
                self.model = tk.layers.Softmax(axis=-1)

            def call(self, x: tf.Tensor, **kwargs):
                action_prob = self.model(self.outputs(x))
                dist = tfd.Categorical(probs=action_prob)
                return action_prob, dist


        class SACDISPolicy(tk.Model):
            def __init__(self,
                         action_space: Space,
                         representation: tk.Model,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(SACDISPolicy, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_critic = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes

                self.actor = ActorNet_SACDIS(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                             normalize, initializer, activation, device)
                self.critic = CriticNet_SACDIS(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                               initializer, activation, device)
                self.target_representation_critic = copy.deepcopy(self.representation_critic)
                self.target_critic = CriticNet_SACDIS(representation.output_shapes['state'][0], self.action_dim,
                                                      critic_hidden_size, initializer, activation, device)
                self.target_critic.set_weights(self.critic.get_weights())

            def call(self, observation: Union[np.ndarray, dict], **kwargs):
                outputs = self.representation(observation)
                act_prob, act_distribution = self.actor(outputs['state'])
                return outputs, act_prob, act_distribution

            def Qtarget(self, observation: Union[np.ndarray, dict]):
                outputs_actor = self.representation(observation)
                outputs_critic = self.target_representation_critic(observation)
                act_prob, act_distribution = self.actor(outputs_actor['state'])
                value = self.target_critic(outputs_critic['state'])
                log_action_prob = tf.math.log(act_prob + 1e-5)
                return act_prob, log_action_prob, value

            def Qaction(self, observation: Union[np.ndarray, dict]):
                outputs_critic = self.representation_critic(observation)
                return outputs_critic, self.critic(outputs_critic['state'])

            def Qpolicy(self, observation: Union[np.ndarray, dict]):
                outputs_actor = self.representation(observation)
                outputs_critic = self.representation_critic(observation)
                act_prob, act_distribution = self.actor(outputs_actor['state'])
                # z = act_prob == 0.0
                # z = z.float() * 1e-8
                log_action_prob = tf.math.log(act_prob + 1e-5)
                return act_prob, log_action_prob, self.critic(outputs_critic['state'])

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.representation_critic.variables, self.target_representation_critic.variables):
                    tp.assign((1 - tau) * tp + tau * ep)
                for ep, tp in zip(self.critic.variables, self.target_critic.variables):
                    tp.assign((1 - tau) * tp + tau * ep)


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.policies import *
        from xuance.mindspore.utils import *
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

            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(ActorNet, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, nn.Softmax, None)[0])
                self.model = nn.SequentialCell(*layers)
                self.sample = self.Sample()
                self.log_prob = self.LogProb()
                self.entropy = self.Entropy()

            def construct(self, x: ms.Tensor):
                return self.model(x)


        class CriticNet(nn.Cell):
            def __init__(self,
                         state_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(CriticNet, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, None)[0])
                self.model = nn.SequentialCell(*layers)

            def construct(self, x: ms.Tensor):
                return self.model(x)[:, 0]


        class ActorCriticPolicy(nn.Cell):
            def __init__(self,
                         action_space: Space,
                         representation: ModuleType,
                         actor_hidden_size: Sequence[int] = None,
                         critic_hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                assert isinstance(action_space, Discrete)
                super(ActorCriticPolicy, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                      normalize, initialize, activation)
                self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                        normalize, initialize, activation)

            def construct(self, observation: ms.tensor):
                outputs = self.representation(observation)
                a = self.actor(outputs['state'])
                v = self.critic(outputs['state'])
                return outputs, a, v


        class ActorPolicy(nn.Cell):
            def __init__(self,
                         action_space: Space,
                         representation: ModuleType,
                         actor_hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                assert isinstance(action_space, Discrete)
                super(ActorPolicy, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                      normalize, initialize, activation)

            def construct(self, observation: ms.tensor):
                outputs = self.representation(observation)
                a = self.actor(outputs['state'])
                return outputs, a


        class PPGActorCritic(nn.Cell):
            def __init__(self,
                         action_space: Space,
                         representation: ModuleType,
                         actor_hidden_size: Sequence[int] = None,
                         critic_hidden_size: Sequence[int] = None,
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None
                         ):
                super(PPGActorCritic, self).__init__()
                self.action_dim = action_space.n
                self.actor_representation = representation
                self.critic_representation = copy.deepcopy(representation)
                self.aux_critic_representation = copy.deepcopy(representation)
                self.representation_info_shape = self.actor_representation.output_shapes

                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                      normalize, initialize, activation)
                self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                        normalize, initialize, activation)
                self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                            normalize, initialize, activation)

            def construct(self, observation: ms.tensor):
                policy_outputs = self.actor_representation(observation)
                critic_outputs = self.critic_representation(observation)
                a = self.actor(policy_outputs['state'])
                v = self.critic(critic_outputs['state'])
                aux_v = self.aux_critic(policy_outputs['state'])
                return policy_outputs, a, v, aux_v

        class CriticNet_SACDIS(nn.Cell):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                super(CriticNet_SACDIS, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], action_dim, None, None, initialize)[0])
                self.model = nn.SequentialCell(*layers)

            def construct(self, x: ms.tensor):
                return self.model(x)


        class SACDISPolicy(nn.Cell):
            def __init__(self,
                         action_space: Space,
                         representation: ModuleType,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                # assert isinstance(action_space, Box)
                super(SACDISPolicy, self).__init__()
                self.action_dim = action_space.n
                self.representation = representation
                self.representation_critic = copy.deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                try:
                    self.representation_params = self.representation.trainable_params()
                except:
                    self.representation_params = []

                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                      normalize, initialize, activation)
                self.critic = CriticNet_SACDIS(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                               initialize, activation)
                self.target_representation_critic = copy.deepcopy(self.representation_critic)
                self.target_critic = copy.deepcopy(self.critic)
                self.actor_params = self.representation_params + self.actor.trainable_params()
                self._log = ms.ops.Log()

            def construct(self, observation: ms.tensor):
                outputs = self.representation(observation)
                act_prob = self.actor(outputs["state"])
                return outputs, act_prob

            def action(self, observation: ms.tensor):
                outputs = self.representation(observation)
                act_prob = self.actor(outputs[0])
                return outputs, act_prob

            def Qtarget(self, observation: ms.tensor):
                outputs = self.representation(observation)
                outputs_critic = self.target_representation_critic(observation)
                act_prob = self.actor(outputs['state'])
                log_action_prob = self._log(act_prob + 1e-10)
                return act_prob, log_action_prob, self.target_critic(outputs_critic['state'])

            def Qaction(self, observation: ms.tensor):
                outputs = self.representation_critic(observation)
                return outputs, self.critic(outputs['state'])

            def Qpolicy(self, observation: ms.tensor):
                outputs = self.representation(observation)
                outputs_critic = self.representation_critic(observation)
                act_prob = self.actor(outputs['state'])
                log_action_prob = self._log(act_prob + 1e-10)
                return act_prob, log_action_prob, self.critic(outputs_critic['state'])

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.representation_critic.trainable_params(), self.target_representation_critic.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
                for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
                    tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
