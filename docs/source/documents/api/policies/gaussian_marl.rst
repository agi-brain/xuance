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
