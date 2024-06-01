import torch
import torch.nn as nn
from copy import deepcopy
from typing import Sequence, Optional, Callable, Union, Dict, List
from gym.spaces import Discrete
from xuance.torch.policies import CategoricalActorNet as ActorNet
from xuance.torch.policies import CategoricalCriticNet as CriticNet
from xuance.torch.policies import VDN_mixer
from xuance.torch.utils import ModuleType, mlp_block, CategoricalDistribution
from xuance.torch import Tensor, Module


class MAAC_Policy(Module):
    """
    MAAC_Policy: Multi-Agent Actor-Critic Policy with categorical policies.
    """

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation_actor: Dict[str, Module],
                 representation_critic: Dict[str, Module],
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.device = device
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.actor_representation = representation_actor
        self.critic_representation = representation_critic

        self.dim_input_critic, self.n_actions = {}, {}
        self.actor, self.critic, self.pi_dist = {}, {}, {}
        for key in self.model_keys:
            self.n_actions[key] = self.action_space[key].n
            dim_obs_actor, dim_obs_critic, dim_act_actor, dim_act_critic = self._get_actor_critic_input(
                self.n_actions[key],
                self.actor_representation[key].output_shapes['state'][0],
                self.critic_representation[key].output_shapes['state'][0], n_agents)

            if self.use_parameter_sharing:
                dim_obs_actor += self.n_agents
                dim_obs_critic += self.n_agents

            self.actor[key] = ActorNet(dim_obs_actor, dim_act_actor, actor_hidden_size,
                                       normalize, initialize, activation, device)
            self.critic[key] = CriticNet(dim_obs_critic, critic_hidden_size, normalize, initialize, activation, device)
            self.pi_dist[key] = CategoricalDistribution(self.n_actions[key])

        self.mixer = mixer

    @property
    def parameters_model(self):
        parameters = [] if self.mixer is None else self.mixer.parameters()
        for key in self.model_keys:
            parameters_key = list(self.actor_representation[key].parameters()) + list(
                self.actor[key].parameters()) + list(self.critic_representation[key].parameters()) + list(
                self.critic[key].parameters())
            parameters += parameters_key
        return parameters

    def _get_actor_critic_input(self, dim_action, dim_actor_rep, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor netwrok and critic networks.

        Parameters:
            dim_action: The dimension of actions.
            dim_actor_rep: The dimension of the output of actor presentation.
            dim_critic_rep: The dimension of the output of critic presentation.
            n_agents: The number of agents.

        Returns:
            dim_actor_in: The dimension of input of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
        """
        dim_actor_in, dim_critic_in = dim_actor_rep, dim_critic_rep
        dim_act_actor, dim_act_critic = dim_action, dim_action
        return dim_actor_in, dim_critic_in, dim_act_actor, dim_act_critic

    def forward(self, observation: Dict[str, Tensor], agent_ids: Tensor = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[str, Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The RNN hidden states of actor representation.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new RNN hidden states of actor representation.
            pi_dists (dict): The stochastic policy distributions.
        """
        rnn_hidden_new, pi_dists = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}

        for key in agent_list:
            if self.use_rnn:
                outputs = self.actor_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.actor_representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                actor_input = torch.concat([outputs['state'], agent_ids], dim=-1)
            else:
                actor_input = outputs['state']

            pi_dists[key] = self.actor[key](actor_input, avail_actions)
        return rnn_hidden_new, pi_dists

    def get_values(self, observation: Dict[str, Tensor], agent_ids: Tensor = None, agent_key: str = None,
                   rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Get critic values via critic networks.

        Parameters:
            observation (Dict[str, Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The RNN hidden states of critic representation.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new RNN hidden states of critic representation.
            values (dict): The evaluated critic values.
        """
        rnn_hidden_new, values = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        for key in agent_list:
            if self.use_rnn:
                outputs = self.critic_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.critic_representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                critic_input = torch.concat([outputs['state'], agent_ids], dim=-1)
            else:
                critic_input = outputs['state']

            values[key] = self.critic[key](critic_input)

        return rnn_hidden, values

    def value_tot(self, values_n: Tensor, global_state=None):
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
                 representation: Module,
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.device = device
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initialize, kwargs['gain'], activation, device)
        self.critic = CriticNet(self.representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                normalize, initialize, activation, device)
        self.mixer = mixer
        self.pi_dist = CategoricalDistribution(self.action_dim)

    def forward(self, observation: Tensor, agent_ids: Tensor,
                *rnn_hidden: Tensor, avail_actions=None, state=None):
        batch_size = len(observation)
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        actor_critic_input = torch.concat([outputs['state'], agent_ids], dim=-1)
        act_logits = self.actor(actor_critic_input)
        if avail_actions is not None:
            avail_actions = Tensor(avail_actions)
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
                values_independent = values_independent.transpose(1, 2).reshape(-1, self.n_agents)
                values_tot = self.value_tot(values_independent, global_state=state)
                values_tot = values_tot.reshape([-1, sequence_length, 1])
                values_tot = values_tot.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
        else:
            values_tot = values_independent if self.mixer is None else self.value_tot(values_independent,
                                                                                      global_state=state)
            values_tot = values_tot.unsqueeze(1).expand(-1, self.n_agents, -1)

        return rnn_hidden, self.pi_dist, values_tot

    def value_tot(self, values_n: Tensor, global_state=None):
        if global_state is not None:
            global_state = torch.as_tensor(global_state).to(self.device)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class COMAPolicy(Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
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
        self.use_rnn = True if kwargs["use_rnn"] else False
        self.actor = ActorNet(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initialize, kwargs['gain'], activation, device)
        critic_input_dim = self.representation.input_shape[0] + self.action_dim * self.n_agents
        if kwargs["use_global_state"]:
            critic_input_dim += kwargs["dim_state"]
        self.critic = COMA_Critic(critic_input_dim, self.action_dim, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.target_critic = deepcopy(self.critic)
        self.parameters_critic = list(self.critic.parameters())
        self.parameters_actor = list(self.representation.parameters()) + list(self.actor.parameters())
        self.pi_dist = CategoricalDistribution(self.action_dim)

    def forward(self, observation: Tensor, agent_ids: Tensor,
                *rnn_hidden: Tensor, avail_actions=None, epsilon=0.0):
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
            avail_actions = Tensor(avail_actions)
            act_probs[avail_actions == 0] = 0.0
        return rnn_hidden, act_probs

    def get_values(self, critic_in: Tensor, *rnn_hidden: Tensor, target=False):
        # get critic values
        v = self.target_critic(critic_in) if target else self.critic(critic_in)
        return [None, None], v

    def copy_target(self):
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(ep)


class MeanFieldActorCriticPolicy(Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs
                 ):
        super(MeanFieldActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor_net = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                  actor_hidden_size, normalize, initialize, kwargs['gain'], activation, device)
        self.critic_net = CriticNet(representation.output_shapes['state'][0] + self.action_dim, n_agents,
                                    critic_hidden_size, normalize, initialize, activation, device)
        self.parameters_actor = list(self.actor_net.parameters()) + list(self.representation.parameters())
        self.parameters_critic = self.critic_net.parameters()
        self.pi_dist = CategoricalDistribution(self.action_dim)

    def forward(self, observation: Tensor, agent_ids: Tensor):
        outputs = self.representation(observation)
        input_actor = torch.concat([outputs['state'], agent_ids], dim=-1)
        act_logits = self.actor_net(input_actor)
        self.pi_dist.set_param(logits=act_logits)
        return outputs, self.pi_dist

    def critic(self, observation: Tensor, actions_mean: Tensor, agent_ids: Tensor):
        outputs = self.representation(observation)
        critic_in = torch.concat([outputs['state'], actions_mean, agent_ids], dim=-1)
        return self.critic_net(critic_in)
