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

**PyTorch:**

.. py:class::
  xuance.torch.policies.categorical.ActorNet(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param action_dim: xxxxxx.
  :type action_dim: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.categorical.ActorNet.forward(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.categorical.CriticNet(state_dim, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.categorical.CriticNet.forward(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.categorical.ActorCriticPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param actor_hidden_size: xxxxxx.
  :type actor_hidden_size: xxxxxx
  :param critic_hidden_size: xxxxxx.
  :type critic_hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.categorical.ActorCriticPolicy.forward(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.categorical.ActorPolicy(action_space, representation, actor_hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param actor_hidden_size: xxxxxx.
  :type actor_hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.categorical.ActorPolicy.forward(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.categorical.PPGActorCritic(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param actor_hidden_size: xxxxxx.
  :type actor_hidden_size: xxxxxx
  :param critic_hidden_size: xxxxxx.
  :type critic_hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.categorical.PPGActorCritic.forward(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.categorical.CriticNet_SACDIS(state_dim, action_dim, hidden_sizes, initialize, activation, device)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param action_dim: xxxxxx.
  :type action_dim: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.categorical.CriticNet_SACDIS.forward(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.categorical.ActorNet_SACDIS(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param action_dim: xxxxxx.
  :type action_dim: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.categorical.ActorNet_SACDIS.forward(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.categorical.SACDISPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param actor_hidden_size: xxxxxx.
  :type actor_hidden_size: xxxxxx
  :param critic_hidden_size: xxxxxx.
  :type critic_hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.categorical.SACDISPolicy.forward(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.categorical.SACDISPolicy.Qtarget(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.categorical.SACDISPolicy.Qaction(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.categorical.SACDISPolicy.Qpolicy(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.categorical.SACDISPolicy.soft_update(tau)

  xxxxxx.

  :param tau: xxxxxx.
  :type tau: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. py:class::
  xuance.mindspore.policies.categorical.ActorNet(state_dim, action_dim, hidden_sizes, normalize, initialize, activation)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param action_dim: xxxxxx.
  :type action_dim: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical.ActorNet.construct(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.categorical.CriticNet(state_dim, hidden_sizes, normalize, initialize, activation)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical.CriticNet.construct(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.categorical.ActorCriticPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param actor_hidden_size: xxxxxx.
  :type actor_hidden_size: xxxxxx
  :param critic_hidden_size: xxxxxx.
  :type critic_hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical.ActorCriticPolicy.construct(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.categorical.ActorPolicy(action_space, representation, actor_hidden_size, normalize, initialize, activation)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param actor_hidden_size: xxxxxx.
  :type actor_hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical.ActorPolicy.construct(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.categorical.PPGActorCritic(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param actor_hidden_size: xxxxxx.
  :type actor_hidden_size: xxxxxx
  :param critic_hidden_size: xxxxxx.
  :type critic_hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical.PPGActorCritic.construct(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.categorical.CriticNet_SACDIS(state_dim, action_dim, hidden_sizes, initialize, activation)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param action_dim: xxxxxx.
  :type action_dim: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical.CriticNet_SACDIS.construct(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.categorical.SACDISPolicy(action_space, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param actor_hidden_size: xxxxxx.
  :type actor_hidden_size: xxxxxx
  :param critic_hidden_size: xxxxxx.
  :type critic_hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical.SACDISPolicy.construct(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical.SACDISPolicy.action(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical.SACDISPolicy.Qtarget(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical.SACDISPolicy.Qaction(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical.SACDISPolicy.Qpolicy(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical.SACDISPolicy.soft_update(tau)

  xxxxxx.

  :param tau: xxxxxx.
  :type tau: xxxxxx

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


        # class SACDISPolicy(nn.Cell):
        #     def __init__(self,
        #                  action_space: Space,
        #                  representation: ModuleType,
        #                  actor_hidden_size: Sequence[int],
        #                  critic_hidden_size: Sequence[int],
        #                  normalize: Optional[ModuleType] = None,
        #                  initialize: Optional[Callable[..., ms.Tensor]] = None,
        #                  activation: Optional[ModuleType] = None):
        #         assert isinstance(action_space, Discrete)
        #         super(SACDISPolicy, self).__init__()
        #         self.action_dim = action_space.n
        #         self.representation = representation
        #         self.representation_info_shape = self.representation.output_shapes
        #         try:
        #             self.representation_params = self.representation.trainable_params()
        #         except:
        #             self.representation_params = []

        #         self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
        #                               normalize, initialize, activation)
        #         self.critic = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
        #                                     initialize, activation)
        #         self.target_actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
        #                                      normalize, initialize, activation)
        #         self.target_critic = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim,
        #                                            critic_hidden_size, initialize, activation)
        #         self.actor_params = self.representation_params + self.actor.trainable_params()
        #         self._unsqueeze = ms.ops.ExpandDims()
        #         self._Categorical = Categorical(dtype=ms.float32)
        #         self.soft_update(tau=1.0)

        #     def action(self, observation: Union[np.ndarray, dict]):
        #         outputs = self.representation(observation)
        #         act_dist = self.actor(outputs[0])
        #         return outputs, act_dist

        #     def Qtarget(self, observation: Union[np.ndarray, dict]):
        #         outputs = self.representation(observation)
        #         act_dist = self.target_actor(outputs['state'])
        #         act = self._Categorical.sample(probs=act_dist)
        #         act_log = self._Categorical.log_prob(value=act, probs=act_dist)
        #         act = self._unsqueeze(act, -1)
        #         return act_log, self.target_critic(outputs['state'], act)

        #     def Qaction(self, observation: Union[np.ndarray, dict], action: ms.Tensor):
        #         outputs = self.representation(observation)
        #         action = self._unsqueeze(action, -1)
        #         return outputs, self.critic(outputs['state'], action)

        #     def Qpolicy(self, observation: Union[np.ndarray, dict]):
        #         outputs = self.representation(observation)
        #         act_dist = self.actor(outputs['state'])
        #         act = self._Categorical.sample(probs=act_dist)
        #         act_log = self._Categorical.log_prob(value=act, probs=act_dist)
        #         act = self._unsqueeze(act, -1)
        #         return act_log, self.critic(outputs['state'], act)

        #     def construct(self):
        #         return super().construct()

        #     def soft_update(self, tau=0.005):
        #         for ep, tp in zip(self.actor.trainable_params(), self.target_actor.trainable_params()):
        #             tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        #         for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
        #             tp.assign_value((tau * ep.data + (1 - tau) * tp.data))

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
