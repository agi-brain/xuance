Gaussian
=======================================

xxxxxx.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.policies.gaussian.ActorNet(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

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
  xuance.torch.policies.gaussian.ActorNet.forward(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.gaussian.CriticNet(state_dim, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.torch.policies.gaussian.CriticNet.forward(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.gaussian.ActorCriticPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

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
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.gaussian.ActorCriticPolicy.forward(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.gaussian.ActorPolicy(action_space, representation, actor_hidden_size, normalize, initialize, activation, device, fixed_std)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str
  :param fixed_std: xxxxxx.
  :type fixed_std: xxxxxx

.. py:function::
  xuance.torch.policies.gaussian.ActorPolicy.forward(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.gaussian.PPGActorCritic(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

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
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.gaussian.PPGActorCritic.forward(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.gaussian.ActorNet_SAC(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

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
  xuance.torch.policies.gaussian.ActorNet_SAC.forward(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.gaussian.CriticNet_SAC(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

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
  xuance.torch.policies.gaussian.CriticNet_SAC.forward(x, a)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :param a: xxxxxx.
  :type a: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.gaussian.SACPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

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
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.gaussian.SACPolicy.forward(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.gaussian.SACPolicy.Qtarget(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.gaussian.SACPolicy.Qaction(observation, action)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: The action input.
  :type action: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.gaussian.SACPolicy.Qpolicy(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.gaussian.SACPolicy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.policies.gaussian.ActorNet(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

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
  xuance.tensorflow.policies.gaussian.ActorNet.call(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.gaussian.CriticNet(state_dim, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
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
  xuance.tensorflow.policies.gaussian.CriticNet.call(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.gaussian.ActorCriticPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

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
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.gaussian.ActorCriticPolicy.call(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.gaussian.ActorPolicy(action_space, representation, actor_hidden_size, normalize, initialize, activation, device, fixed_std)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str
  :param fixed_std: xxxxxx.
  :type fixed_std: xxxxxx

.. py:function::
  xuance.tensorflow.policies.gaussian.ActorPolicy.call(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.gaussian.PPGActorCritic(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

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
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.gaussian.PPGActorCritic.call(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.gaussian.ActorNet_SAC(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

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
  xuance.tensorflow.policies.gaussian.ActorNet_SAC.call(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.gaussian.CriticNet_SAC(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

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
  xuance.tensorflow.policies.gaussian.CriticNet_SAC.call(inputs)

  xxxxxx.

  :param inputs: The inputs of the neural neworks.
  :type inputs: Dict(tf.Tensor)
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.gaussian.SACPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

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
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.gaussian.SACPolicy.call(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.gaussian.SACPolicy.Qtarget(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.gaussian.SACPolicy.Qaction(observation, action)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :param action: The action input.
  :type action: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.gaussian.SACPolicy.Qpolicy(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.gaussian.SACPolicy.soft_update(tau)

  xxxxxx.

  :param tau: The soft update factor for the update of target networks.
  :type tau: float
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.policies.gaussian.ActorNet(state_dim, action_dim, hidden_sizes, normalize, initialize, activation)

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
  xuance.mindspore.policies.gaussian.ActorNet.construct(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.gaussian.CriticNet(state_dim, hidden_sizes, normalize, initialize, activation)

  :param state_dim: The dimension of the input state.
  :type state_dim: int
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.gaussian.CriticNet.construct(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.gaussian.ActorCriticPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

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
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.gaussian.ActorCriticPolicy.construct(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.gaussian.ActorPolicy(action_space, representation, actor_hidden_size, normalize, initialize, activation)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.gaussian.ActorPolicy.construct(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.gaussian.ActorNet_SAC(state_dim, action_dim, hidden_sizes, initialize, activation)

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

.. py:function::
  xuance.mindspore.policies.gaussian.ActorNet_SAC.construct(x)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.gaussian.CriticNet_SAC(state_dim, action_dim, hidden_sizes, initialize, activation)

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

.. py:function::
  xuance.mindspore.policies.gaussian.CriticNet_SAC.construct(x, a)

  xxxxxx.

  :param x: The input tensor.
  :type x: torch.Tensor
  :param a: xxxxxx.
  :type a: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.gaussian.SACPolicy(action_space, representation, actor_hidden_size, initialize, activation)

  :param action_space: The action space of the environment.
  :type action_space: Space
  :param representation: The representation module.
  :type representation: nn.Module
  :param actor_hidden_size: The sizes of the hidden layers in actor network.
  :type actor_hidden_size: list
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module

.. py:function::
  xuance.mindspore.policies.gaussian.SACPolicy.action(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian.SACPolicy.Qtarget(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian.SACPolicy.Qaction(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian.SACPolicy.Qpolicy(observation)

  xxxxxx.

  :param observation: The original observation variables.
  :type observation: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian.SACPolicy.construct()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian.SACPolicy.soft_update(tau)

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

        from xuance.torch.policies import *
        from xuance.torch.utils import *
        from xuance.torch.representations import Basic_Identical


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
                self.mu = nn.Sequential(*layers)
                self.logstd = nn.Parameter(-torch.ones((action_dim,), device=device))
                self.dist = DiagGaussianDistribution(action_dim)

            def forward(self, x: torch.Tensor):
                self.dist.set_param(self.mu(x), self.logstd.exp())
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
                layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
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
                self.action_dim = action_space.shape[0]
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
                         device: Optional[Union[str, int, torch.device]] = None,
                         fixed_std: bool = True):
                super(ActorPolicy, self).__init__()
                self.action_dim = action_space.shape[0]
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
                self.action_dim = action_space.shape[0]
                self.actor_representation = representation
                self.critic_representation = copy.deepcopy(representation)
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
                a = self.actor(policy_outputs['state'])
                v = self.critic(critic_outputs['state'])
                aux_v = self.aux_critic(policy_outputs['state'])
                return policy_outputs, a, v, aux_v


        class ActorNet_SAC(nn.Module):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(ActorNet_SAC, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
                    layers.extend(mlp)
                self.device = device
                self.output = nn.Sequential(*layers)
                self.out_mu = nn.Sequential(nn.Linear(hidden_sizes[-1], action_dim, device=device), nn.Tanh())
                self.out_std = nn.Linear(hidden_sizes[-1], action_dim, device=device)

            def forward(self, x: torch.tensor):
                output = self.output(x)
                mu = self.out_mu(output)
                # std = torch.tanh(self.out_std(output))
                std = torch.clamp(self.out_std(output), -20, 2)
                std = std.exp()
                # dia_std = torch.diag_embed(std)
                self.dist = torch.distributions.Normal(mu, std)
                return self.dist


        class CriticNet_SAC(nn.Module):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(CriticNet_SAC, self).__init__()
                layers = []
                input_shape = (state_dim + action_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.tensor, a: torch.tensor):
                return self.model(torch.concat((x, a), dim=-1))[:, 0]


        class SACPolicy(nn.Module):
            def __init__(self,
                         action_space: Space,
                         representation: nn.Module,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         device: Optional[Union[str, int, torch.device]] = None):
                super(SACPolicy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.representation_info_shape = representation.output_shapes
                self.representation_actor = representation
                self.representation_critic = copy.deepcopy(representation)
                self.actor = ActorNet_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                          normalize, initialize, activation, device)
                self.critic = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                            normalize, initialize, activation, device)

                self.target_representation_actor = copy.deepcopy(self.representation_actor)
                self.target_actor = copy.deepcopy(self.actor)
                self.target_representation_critic = copy.deepcopy(self.representation_critic)
                self.target_critic = copy.deepcopy(self.critic)

            def forward(self, observation: Union[np.ndarray, dict]):
                outputs_actor = self.representation_actor(observation)
                act_dist = self.actor(outputs_actor['state'])
                return outputs_actor, act_dist

            def Qtarget(self, observation: Union[np.ndarray, dict]):
                outputs_actor = self.target_representation_actor(observation)
                outputs_critic = self.target_representation_critic(observation)
                act_dist = self.target_actor(outputs_actor['state'])
                act = act_dist.rsample()
                act_log = act_dist.log_prob(act).sum(-1)
                return act_log, self.target_critic(outputs_critic['state'], act)

            def Qaction(self, observation: Union[np.ndarray, dict], action: torch.Tensor):
                outputs_critic = self.representation_critic(observation)
                return self.critic(outputs_critic['state'], action)

            def Qpolicy(self, observation: Union[np.ndarray, dict]):
                outputs_actor = self.representation_actor(observation)
                outputs_critic = self.representation_critic(observation)
                act_dist = self.actor(outputs_actor['state'])
                act = act_dist.rsample()
                act_log = act_dist.log_prob(act).sum(-1)
                return act_log, self.critic(outputs_critic['state'], act)

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.representation_actor.parameters(), self.target_representation_actor.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.representation_critic.parameters(), self.target_representation_critic.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)
                for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * ep.data)



  .. group-tab:: TensorFlow

    .. code-block:: python

        from xuance.tensorflow.policies import *
        from xuance.tensorflow.utils import *
        from xuance.tensorflow.representations import Basic_Identical
        import tensorflow_probability as tfp

        tfd = tfp.distributions


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
                self.mu_model = tk.Sequential(layers)
                self.logstd = tf.Variable(tf.zeros((action_dim,)) - 1, trainable=True)
                self.dist = DiagGaussianDistribution(action_dim)

            def call(self, x: tf.Tensor, **kwargs):
                self.dist.set_param(self.mu_model(x), tf.math.exp(self.logstd))
                return self.mu_model(x)


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
                self.action_dim = action_space.shape[0]
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
                         device: str = "cpu:0",
                         fixed_std: bool = True):
                super(ActorPolicy, self).__init__()
                self.action_dim = action_space.shape[0]
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
                super(PPGActorCritic, self).__init__()
                self.action_dim = action_space.shape[0]
                self.actor_representation = representation
                self.critic_representation = copy.deepcopy(representation)
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
                a = self.actor(policy_outputs['state'])
                v = self.critic(critic_outputs['state'])
                aux_v = self.aux_critic(policy_outputs)
                return policy_outputs, a, v, aux_v


        class ActorNet_SAC(tk.Model):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(ActorNet_SAC, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initializer, device)
                    layers.extend(mlp)
                self.device = device
                self.outputs = tk.Sequential(layers)
                self.out_mu = tk.layers.Dense(units=action_dim,
                                              input_shape=(hidden_sizes[0],))
                self.out_std = tk.layers.Dense(units=action_dim,
                                               input_shape=(hidden_sizes[0],))

            def call(self, x: tf.Tensor, **kwargs):
                output = self.outputs(x)
                mu = tf.tanh(self.out_mu(output))
                std = tf.clip_by_value(self.out_std(output), -20, 2)
                std = tf.exp(std)
                return tfd.Normal(mu, std)
                # self.dist = tfd.Normal(mu, std)
                # return mu, std


        class CriticNet_SAC(tk.Model):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                super(CriticNet_SAC, self).__init__()
                layers = []
                input_shape = (state_dim + action_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initializer, device)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initializer, device)[0])
                self.model = tk.Sequential(layers)

            def call(self, inputs: Union[np.ndarray, dict], **kwargs):
                obs = inputs['obs']
                act = inputs['act']
                return self.model(tf.concat((obs, act), axis=-1))


        class SACPolicy(tk.Model):
            def __init__(self,
                         action_space: Space,
                         representation: Basic_Identical,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu:0"):
                assert isinstance(action_space, Box)
                super(SACPolicy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes

                self.actor = ActorNet_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                          initializer, activation, device)
                self.critic = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                            initializer, activation, device)
                self.target_actor = ActorNet_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                                 initializer, activation, device)
                self.target_critic = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim,
                                                   critic_hidden_size,
                                                   initializer, activation, device)
                self.soft_update(tau=1.0)

            def action(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                dist = self.actor(outputs['state'])

                return outputs, dist

            def Qtarget(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                act_dist = self.target_actor(outputs['state'])
                act = act_dist.sample()
                act_log = act_dist.log_prob(act)
                inputs = {'obs': outputs['state'], 'act': act}
                return outputs, act_log, self.target_critic(inputs)

            def Qaction(self, observation: Union[np.ndarray, dict], action: tf.Tensor):
                outputs = self.representation(observation)
                inputs = {'obs': outputs['state'], 'act': action}
                return outputs, self.critic(inputs)

            def Qpolicy(self, observation: Union[np.ndarray, dict]):
                outputs = self.representation(observation)
                act_dist = self.actor(outputs['state'])
                act = act_dist.sample()
                act_log = act_dist.log_prob(act)
                inputs = {'obs': outputs['state'], 'act': act}
                return outputs, act_log, self.critic(inputs)

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor.variables, self.target_actor.variables):
                    tp.assign((1 - tau) * tp + tau * ep)
                for ep, tp in zip(self.critic.variables, self.target_critic.variables):
                    tp.assign((1 - tau) * tp + tau * ep)


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.policies import *
        from xuance.mindspore.utils import *
        from mindspore.nn.probability.distribution import Normal
        import copy

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
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                super(ActorNet, self).__init__()
                layers = []
                input_shape = (state_dim,)
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

            def construct(self, x: ms.Tensor):
                return self.mu(x)


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
                assert isinstance(action_space, Box)
                super(ActorCriticPolicy, self).__init__()
                self.action_dim = action_space.shape[0]
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
                         activation: Optional[ModuleType] = None):
                assert isinstance(action_space, Box)
                super(ActorPolicy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                      normalize, initialize, activation)

            def construct(self, observation: ms.tensor):
                outputs = self.representation(observation)
                a = self.actor(outputs['state'])
                return outputs, a


        class ActorNet_SAC(nn.Cell):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                super(ActorNet_SAC, self).__init__()
                layers = []
                input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize)
                    layers.extend(mlp)
                self.output = nn.SequentialCell(*layers)
                self.out_mu = nn.Dense(hidden_sizes[0], action_dim)
                self.out_std = nn.Dense(hidden_sizes[0], action_dim)
                self._tanh = ms.ops.Tanh()
                self._exp = ms.ops.Exp()

            def construct(self, x: ms.tensor):
                output = self.output(x)
                mu = self._tanh(self.out_mu(output))
                std = ms.ops.clip_by_value(self.out_std(output), -20, 2)
                std = self._exp(std)
                # dist = Normal(mu, std)
                # return dist
                return mu, std


        class CriticNet_SAC(nn.Cell):
            def __init__(self,
                         state_dim: int,
                         action_dim: int,
                         hidden_sizes: Sequence[int],
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                super(CriticNet_SAC, self).__init__()
                layers = []
                input_shape = (state_dim + action_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize)
                    layers.extend(mlp)
                layers.extend(mlp_block(input_shape[0], 1, None, None, initialize)[0])
                self.model = nn.SequentialCell(*layers)
                self._concat = ms.ops.Concat(-1)

            def construct(self, x: ms.tensor, a: ms.tensor):
                return self.model(self._concat((x, a)))[:, 0]


        class SACPolicy(nn.Cell):
            def __init__(self,
                         action_space: Space,
                         representation: ModuleType,
                         actor_hidden_size: Sequence[int],
                         critic_hidden_size: Sequence[int],
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None):
                assert isinstance(action_space, Box)
                super(SACPolicy, self).__init__()
                self.action_dim = action_space.shape[0]
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                try:
                    self.representation_params = self.representation.trainable_params()
                except:
                    self.representation_params = []

                self.actor = ActorNet_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                          initialize, activation)
                self.critic = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                            initialize, activation)
                self.target_actor = copy.deepcopy(self.actor)
                self.target_critic = copy.deepcopy(self.critic)
                self.nor = Normal()

            def action(self, observation: ms.tensor):
                outputs = self.representation(observation)
                # act_dist = self.actor(outputs[0])
                mu, std = self.actor(outputs['state'])
                act_dist = Normal(mu, std)

                return outputs, act_dist

            def Qtarget(self, observation: ms.tensor):
                outputs = self.representation(observation)
                # act_dist = self.target_actor(outputs[0])
                mu, std = self.target_actor(outputs['state'])
                # act_dist = Normal(mu, std)

                # act = act_dist.sample()
                # act_log = act_dist.log_prob(act)
                act = self.nor.sample(mean=mu, sd=std)
                act_log = self.nor.log_prob(act, mu, std)
                return outputs, act_log, self.target_critic(outputs['state'], act)

            def Qaction(self, observation: ms.tensor, action: ms.Tensor):
                outputs = self.representation(observation)
                return outputs, self.critic(outputs['state'], action)

            def Qpolicy(self, observation: ms.tensor):
                outputs = self.representation(observation)
                # act_dist = self.actor(outputs['state'])
                mu, std = self.actor(outputs['state'])
                # act_dist = Normal(mu, std)

                # act = act_dist.sample()
                # act_log = act_dist.log_prob(act)
                act = self.nor.sample(mean=mu, sd=std)
                act_log = self.nor.log_prob(act, mu, std)
                return outputs, act_log, self.critic(outputs['state'], act)

            def construct(self):
                return super().construct()

            def soft_update(self, tau=0.005):
                for ep, tp in zip(self.actor.trainable_params(), self.target_actor.trainable_params()):
                    tp.assign_value((tau*ep.data+(1-tau)*tp.data))
                for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
                    tp.assign_value((tau*ep.data+(1-tau)*tp.data))