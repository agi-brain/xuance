Gaussian-MARL
=======================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.policies.gaussian_marl.BasicQhead(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param action_dim: xxxxxx.
  :type action_dim: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
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
  xuance.torch.policies.gaussian_marl.BasicQhead.forward(x)

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx


.. py:class::
  xuance.torch.policies.gaussian_marl.BasicQnetwork(action_space, n_agents, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
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
  xuance.torch.policies.gaussian_marl.BasicQnetwork.forward(observation, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.gaussian_marl.BasicQnetwork.target_Q(observation, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.gaussian_marl.BasicQnetwork.copy_target()

  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.gaussian_marl.ActorNet(state_dim, n_agents, action_dim, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
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
  xuance.torch.policies.gaussian_marl.ActorNet.forward(x)

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.gaussian_marl.CriticNet(state_dim, n_agents, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
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
  xuance.torch.policies.gaussian_marl.CriticNet.forward(x)

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.gaussian_marl.MAAC_Policy(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param mixer: xxxxxx.
  :type mixer: xxxxxx
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
  xuance.torch.policies.gaussian_marl.MAAC_Policy.forward(observation, agent_ids, *rnn_hidden)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param *rnn_hidden: xxxxxx.
  :type *rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.gaussian_marl.MAAC_Policy.get_values(critic_in, agent_ids, *rnn_hidden)

  :param critic_in: xxxxxx.
  :type critic_in: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param *rnn_hidden: xxxxxx.
  :type *rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.gaussian_marl.MAAC_Policy.value_tot(values_n, global_state)

  :param values_n: xxxxxx.
  :type values_n: xxxxxx
  :param global_state: xxxxxx.
  :type global_state: xxxxxx
  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.gaussian_marl.Basic_ISAC_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
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
  xuance.torch.policies.gaussian_marl.Basic_ISAC_policy.forward(observation, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.gaussian_marl.Basic_ISAC_policy.critic(observation, actions, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.gaussian_marl.Basic_ISAC_policy.target_critic(observation, actions, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: None.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.gaussian_marl.Basic_ISAC_policy.target_actor(observation, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: None.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.gaussian_marl.Basic_ISAC_policy.soft_update(tau)

  :param tau: xxxxxx.
  :type tau: xxxxxx
  :return: None.
  :rtype: xxxxxx

.. py:class::
 xuance.torch.policies.gaussian_marl.MASAC_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
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
  xuance.torch.policies.gaussian_marl.MASAC_policy.critic(observation, actions, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.gaussian_marl.MASAC_policy.target_critic(observation, actions, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. py:class::
  xuance.mindspore.policies.gaussian_marl.BasicQhead(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param action_dim: xxxxxx.
  :type action_dim: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.BasicQhead.construct(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.gaussian_marl.BasicQnetwork(action_space, n_agents, representation, hidden_sizes, normalize, initialize, activation)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.BasicQnetwork.construct(observation, agent_ids)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.BasicQnetwork.target_Q(observation, agent_ids)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.BasicQnetwork.copy_target()

  xxxxxx.

.. py:class::
  xuance.mindspore.policies.gaussian_marl.ActorNet(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param action_dim: xxxxxx.
  :type action_dim: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.ActorNet.construct(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.gaussian_marl.CriticNet(state_dim, n_agents, hidden_sizes, normalize, initialize, activation)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.CriticNet.construct(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.gaussian_marl.MAAC_Policy(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, kwargs)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param mixer: xxxxxx.
  :type mixer: xxxxxx
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
  :param kwargs: xxxxxx.
  :type kwargs: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MAAC_Policy.construct(observation, agent_ids, rnn_hidden, kwargs)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param rnn_hidden: xxxxxx.
  :type rnn_hidden: xxxxxx
  :param kwargs: xxxxxx.
  :type kwargs: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MAAC_Policy.get_values(observation, agent_ids, rnn_hidden, kwargs)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param rnn_hidden: xxxxxx.
  :type rnn_hidden: xxxxxx
  :param kwargs: xxxxxx.
  :type kwargs: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MAAC_Policy.value_tot(values_n, global_state)

  xxxxxx.

  :param values_n: xxxxxx.
  :type values_n: xxxxxx
  :param global_state: xxxxxx.
  :type global_state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.gaussian_marl.Basic_ISAC_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
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
  xuance.mindspore.policies.gaussian_marl.Basic_ISAC_policy.construct(observation, agent_ids)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.Basic_ISAC_policy.critic(observation, actions, agent_ids)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.Basic_ISAC_policy.critic_for_train(observation, actions, agent_ids)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.Basic_ISAC_policy.target_critic(observation, actions, agent_ids)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.Basic_ISAC_policy.target_actor(observation, agent_ids)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.Basic_ISAC_policy.soft_update(tau)

  xxxxxx.

  :param tau: xxxxxx.
  :type tau: xxxxxx

.. py:class::
  xuance.mindspore.policies.gaussian_marl.MASAC_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
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
  xuance.mindspore.policies.gaussian_marl.MASAC_policy.construct(observation, agent_ids)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MASAC_policy.critic(observation, actions, agent_ids)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MASAC_policy.critic_for_train(observation, actions, agent_ids)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MASAC_policy.target_critic(observation, actions, agent_ids)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MASAC_policy.target_actor(observation, agent_ids)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.gaussian_marl.MASAC_policy.soft_update(tau)

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
                         initialize: Optional[Callable[..., torch.Tensor]] = None,
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

