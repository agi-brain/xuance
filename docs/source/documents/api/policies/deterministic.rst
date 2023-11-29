Deterministic
====================================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.policies.deterministic.BasicQhead(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

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
  xuance.torch.policies.deterministic.BasicQhead.forward(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.BasicRecurrent(**kwargs)

  :param **kwargs: xxxxxx.
  :type **kwargs: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.BasicRecurrent.forward(x, h, c)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :param h: xxxxxx.
  :type h: xxxxxx
  :param c: xxxxxx.
  :type c: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.DuelQhead(state_dim, action_dim, hidden_sizes, normalize, initialize, activation, device)

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
  xuance.torch.policies.deterministic.DuelQhead.forward(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.C51Qhead(state_dim, action_dim, atom_num, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param action_dim: xxxxxx.
  :type action_dim: xxxxxx
  :param atom_num: xxxxxx.
  :type atom_num: xxxxxx
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
  xuance.torch.policies.deterministic.C51Qhead.forward(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.QRDQNhead(state_dim, action_dim, atom_num, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param action_dim: xxxxxx.
  :type action_dim: xxxxxx
  :param atom_num: xxxxxx.
  :type atom_num: xxxxxx
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
  xuance.torch.policies.deterministic.QRDQNhead.forward(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.BasicQnetwork(action_space, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.BasicQnetwork.forward(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.BasicQnetwork.target(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.BasicQnetwork.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.DuelQnetwork(action_space, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.DuelQnetwork.forward(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.DuelQnetwork.target(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.DuelQnetwork.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.NoisyQnetwork(action_space, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

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

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.NoisyQnetwork.target(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.NoisyQnetwork.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.C51Qnetwork(action_space, atom_num, vmin, vmax, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param atom_num: xxxxxx.
  :type atom_num: xxxxxx
  :param vmin: xxxxxx.
  :type vmin: xxxxxx
  :param vmax: xxxxxx.
  :type vmax: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.C51Qnetwork.forward(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.C51Qnetwork.target(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.C51Qnetwork.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic.QRDQN_Network(action_space, quantile_num, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param quantile_num: xxxxxx.
  :type quantile_num: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.QRDQN_Network.forward(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.QRDQN_Network.target(observation)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic.QRDQN_Network.copy_target()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

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
