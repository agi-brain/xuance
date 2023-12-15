Categorical-MARL
======================================


.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.policies.categorical_mal.ActorNet(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, gain, activation, device)

  :param state_dim: The dimention of the state varibale.
  :type state_dim: int
  :param action_dim: The dimentin of the actions.
  :type action_dim: int
  :param n_agents: The number of all agents.
  :type n_agents: int
  :param hidden_sizes: The number of hidden units for hidden layers.
  :type hidden_sizes: list
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param gain: xxxxxx.
  :type gain: xxxxxx.
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.categorical_mal.ActorNet.forward(x)

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.categorical_mal.CriticNet(state_dim, n_agents, hidden_sizes,  normalize, initialize, activation, device)

  :param state_dim: The dimention of the state varibale.
  :type state_dim: int
  :param n_agents: number of agents.
  :type n_agents: int
  :param hidden_sizes: The number of hidden units for hidden layers.
  :type hidden_sizes: list
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.categorical_mal.CriticNet.forward(x)

  :param x: input data.
  :type x: torch.Tensor
  :return: y.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.categorical_mal.COMA_Critic(state_dim, act_dim, hidden_sizes,  normalize, initialize, activation, device)

  :param state_dim: The dimention of the state varibale.
  :type state_dim: int
  :param act_dim: The dimention of actions.
  :type act_dim: int
  :param hidden_sizes: The number of hidden units for hidden layers.
  :type hidden_sizes: list
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.categorical_mal.COMA_Critic.forward(x)

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.categorical_mal.MAAC_Policy(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: The action space.
  :type action_space: xxxxxx
  :param n_agents: The number of agents.
  :type n_agents: int
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param mixer: xxxxxx.
  :type mixer: xxxxxx
  :param actor_hidden_size: The number of hidden units for actor's hidden layers.
  :type actor_hidden_size: list
  :param critic_hidden_size: The number of hidden units for critic's hidden layers.
  :type critic_hidden_size: list
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.categorical_mal.MAAC_Policy.forward(observation, agent_ids, *rnn_hidden, avail_actions)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param *rnn_hidden: xxxxxx.
  :type *rnn_hidden: xxxxxx
  :param avail_actions: xxxxxx.
  :type avail_actions: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.categorical_mal.MAAC_Policy.get_values(critic_in, agent_ids, *rnn_hidden)

  :param critic_in: xxxxxx.
  :type critic_in: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param *rnn_hidden: xxxxxx.
  :type *rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.categorical_mal.MAAC_Policy.value_tot(values_n, global_state)

  :param values_n: xxxxxx.
  :type values_n: xxxxxx
  :param global_state: xxxxxx.
  :type global_state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.categorical_mal.MAAC_Policy_Share(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

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
  xuance.torch.policies.categorical_mal.MAAC_Policy_Share.forward(observation, agent_ids, *rnn_hidden, avail_actions)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param *rnn_hidden: xxxxxx.
  :type *rnn_hidden: xxxxxx
  :param avail_actions: xxxxxx.
  :type avail_actions: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.categorical_mal.MAAC_Policy_Share.value_tot(values_n, global_state)

  :param values_n: xxxxxx.
  :type values_n: xxxxxx
  :param global_state: xxxxxx.
  :type global_state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.categorical_mal.COMAPolicy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

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
  xuance.torch.policies.categorical_mal.COMAPolicy.forward(observation, agent_ids, *rnn_hidden, avail_actions)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param *rnn_hidden: xxxxxx.
  :type *rnn_hidden: xxxxxx
  :param avail_actions: xxxxxx.
  :type avail_actions: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.categorical_mal.COMAPolicy.get_values(critic_in, *rnn_hidden, target)

  :param critic_in: xxxxxx.
  :type critic_in: xxxxxx
  :param *rnn_hidden: xxxxxx.
  :type *rnn_hidden: xxxxxx
  :param target: xxxxxx.
  :type target: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.categorical_mal.COMAPolicy.copy_target()

  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.categorical_mal.MeanFieldActorCriticPolicy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

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
  xuance.torch.policies.categorical_mal.MeanFieldActorCriticPolicy.forward(observation, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.categorical_mal.MeanFieldActorCriticPolicy.target_actor(observation, agent_ids)

 :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.categorical_mal.MeanFieldActorCriticPolicy.critic(observation, actions_mean, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions_mean: xxxxxx.
  :type actions_mean: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.categorical_mal.MeanFieldActorCriticPolicy.target_critic(observation, actions_mean, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions_mean: xxxxxx.
  :type actions_mean: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.categorical_mal.MeanFieldActorCriticPolicy.soft_update(tau)

  :param tau: xxxxxx.
  :type tau: xxxxxx
  :return: NONE.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. py:class::
  xuance.mindspore.policies.categorical_marl.ActorNet(state_dim, action_dim, n_agents, hidden_sizes, normalize, initialize, activation)

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
  xuance.mindspore.policies.categorical_marl.ActorNet.construct(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.categorical_marl.CriticNet(state_dim, n_agents, hidden_sizes, normalize, initialize, activation)

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
  xuance.mindspore.policies.categorical_marl.CriticNet.construct(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.categorical_marl.COMA_Critic(state_dim, act_dim, hidden_sizes, normalize, initialize, activation)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param act_dim: xxxxxx.
  :type act_dim: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical_marl.COMA_Critic.construct(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.categorical_marl.MAAC_Policy(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

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

.. py:function::
  xuance.mindspore.policies.categorical_marl.MAAC_Policy.construct(observation, agent_ids, rnn_hidden, avail_actions)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param rnn_hidden: xxxxxx.
  :type rnn_hidden: xxxxxx
  :param avail_actions: xxxxxx.
  :type avail_actions: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical_marl.MAAC_Policy.get_values(critic_in, agent_ids, rnn_hidden)

  xxxxxx.

  :param critic_in: xxxxxx.
  :type critic_in: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param rnn_hidden: xxxxxx.
  :type rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical_marl.MAAC_Policy.value_tot(values_n, global_state)

  xxxxxx.

  :param values_n: xxxxxx.
  :type values_n: xxxxxx
  :param global_state: xxxxxx.
  :type global_state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.categorical_marl.MAAC_Policy_Share(action_space, n_agents, representation, mixer, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

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
  xuance.mindspore.policies.categorical_marl.MAAC_Policy_Share.construct(observation, agent_ids, rnn_hidden, avail_actions)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param rnn_hidden: xxxxxx.
  :type rnn_hidden: xxxxxx
  :param avail_actions: xxxxxx.
  :type avail_actions: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical_marl.MAAC_Policy_Share.value_tot(values_n, global_state)

  xxxxxx.

  :param values_n: xxxxxx.
  :type values_n: xxxxxx
  :param global_state: xxxxxx.
  :type global_state: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.categorical_marl.COMAPolicy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

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
  xuance.mindspore.policies.categorical_marl.COMAPolicy.construct(observation, agent_ids, rnn_hidden, avail_actions, epsilon)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param rnn_hidden: xxxxxx.
  :type rnn_hidden: xxxxxx
  :param avail_actions: xxxxxx.
  :type avail_actions: xxxxxx
  :param epsilon: xxxxxx.
  :type epsilon: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical_marl.COMAPolicy.get_values(critic_in, rnn_hidden, target)

  xxxxxx.

  :param critic_in: xxxxxx.
  :type critic_in: xxxxxx
  :param rnn_hidden: xxxxxx.
  :type rnn_hidden: xxxxxx
  :param target: xxxxxx.
  :type target: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical_marl.COMAPolicy.copy_target()

  xxxxxx.

.. py:class::
  xuance.mindspore.policies.categorical_marl.MeanFieldActorCriticPolicy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation)

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
  xuance.mindspore.policies.categorical_marl.MeanFieldActorCriticPolicy.construct(observation, agent_ids)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.categorical_marl.MeanFieldActorCriticPolicy.get_values(observation, actions_mean, agent_ids)

  xxxxxx.

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions_mean: xxxxxx.
  :type actions_mean: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

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


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.policies import *
        from xuance.mindspore.utils import *
        from xuance.mindspore.representations import Basic_Identical
        from .deterministic_marl import BasicQhead
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
                          *rnn_hidden: torch.Tensor, avail_actions=None):
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
                                      actor_hidden_size, normalize, initialize, kwargs['gain'], activation)
                self.critic = CriticNet(self.representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                        normalize, initialize, activation)
                self.mixer = mixer
                self._concat = ms.ops.Concat(axis=-1)
                self.expand_dims = ms.ops.ExpandDims()
                self._softmax = nn.Softmax(axis=-1)

            def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                          *rnn_hidden: torch.Tensor, avail_actions=None, state=None):
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

            def get_values(self, critic_in: torch.Tensor, *rnn_hidden: torch.Tensor, target=False):
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
