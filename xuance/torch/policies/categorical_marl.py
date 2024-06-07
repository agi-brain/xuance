import torch
import torch.nn as nn
from copy import deepcopy
from gym.spaces import Discrete
from typing import Sequence, Optional, Callable, Union, Dict, List
from xuance.torch.policies import CategoricalActorNet, ActorNet
from xuance.torch.policies.core import CriticNet
from xuance.torch.policies import VDN_mixer
from xuance.torch.utils import ModuleType, CategoricalDistribution
from xuance.torch import Tensor, Module, ModuleDict


class MAAC_Policy(Module):
    """
    MAAC_Policy: Multi-Agent Actor-Critic Policy with categorical policies.
    """

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation_actor: ModuleDict,
                 representation_critic: ModuleDict,
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
        self.actor, self.critic = ModuleDict(), ModuleDict()
        for key in self.model_keys:
            self.n_actions[key] = self.action_space[key].n
            dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out = self._get_actor_critic_input(
                self.n_actions[key],
                self.actor_representation[key].output_shapes['state'][0],
                self.critic_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = CategoricalActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                                  normalize, initialize, activation, device)
            self.critic[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation, device)

        self.mixer = mixer

    @property
    def parameters_model(self):
        parameters = list(self.actor_representation.parameters()) + list(self.actor.parameters()) + list(
            self.critic_representation.parameters()) + list(self.critic.parameters())
        if self.mixer is None:
            return parameters
        else:
            return parameters + list(self.mixer.parameters())

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
            dim_actor_out: The dimension of output of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
            dim_critic_out: The dimension of the output of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in, dim_critic_out = dim_critic_rep, dim_action
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out

    def forward(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
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

            avail_actions_input = None if avail_actions is None else avail_actions[key]
            pi_dists[key] = self.actor[key](actor_input, avail_actions_input)
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

        return rnn_hidden_new, values

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
        self.actor = CategoricalActorNet(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
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


class COMA_Policy(Module):
    """
    COMA_Policy: Counterfactual Multi-Agent Actor-Critic Policy with categorical distributions.
    """

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation_actor: Module,
                 representation_critic: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(COMA_Policy, self).__init__()
        self.device = device
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.use_global_state = kwargs['use_global_state']
        self.model_keys = kwargs['model_keys']
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.actor_representation = representation_actor
        self.critic_representation = representation_critic
        self.target_critic_representation = deepcopy(self.critic_representation)

        # create actor
        self.actor = ModuleDict()
        self.pi_dist, self.n_actions = {}, {}
        dim_critic_input = {}
        for key in self.model_keys:
            self.n_actions[key] = self.action_space[key].n
            dim_actor_input = self.actor_representation[key].output_shapes['state'][0]
            dim_critic_input[key] = self.critic_representation[key].output_shapes['state'][0]
            if self.use_parameter_sharing:
                dim_actor_input += self.n_agents
            self.actor[key] = ActorNet(dim_actor_input, self.n_actions[key], actor_hidden_size,
                                       normalize, initialize, activation, None, device)
            self.pi_dist[key] = CategoricalDistribution(self.n_actions[key])

        critic_input_dim = sum(dim_critic_input.values()) + sum(self.n_actions.values())
        if kwargs["use_global_state"]:
            critic_input_dim += kwargs["dim_state"]
        self.critic = CriticNet(critic_input_dim, critic_hidden_size, normalize, initialize, activation, device)
        self.target_critic = deepcopy(self.critic)

    @property
    def parameters_actor(self):
        return list(self.actor_representation.parameters()) + list(self.actor.parameters())

    @property
    def parameters_critic(self):
        return list(self.critic_representation.parameters()) + list(self.critic.parameters())

    def forward(self,
                observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, epsilon=0.0):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[str, Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The RNN hidden states of actor representation.
            epsilon: The epsilon.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new RNN hidden states of actor representation.
            act_probs (dict): The probabilities of the actions.
        """
        rnn_hidden_new, pi_logits, act_probs = {}, {}, {}
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

            pi_logits[key] = self.actor[key](actor_input, avail_actions)
            act_probs[key] = nn.functional.softmax(pi_logits[key], dim=-1)
            act_probs[key] = (1 - epsilon) * act_probs[key] + epsilon * 1 / self.n_actions[key]
            if avail_actions is not None:
                avail_actions = Tensor(avail_actions)
                act_probs[key][avail_actions == 0] = 0.0

        return rnn_hidden_new, act_probs

    def get_values(self, observation: Dict[str, Tensor], actions: Dict[str, Tensor], state: Optional[Tensor] = None,
                   agent_ids: Tensor = None, rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, target=False):
        """
        Get evaluated critic values.

        Parameters:
            observation (Dict[str, Tensor]): The input observations for the policies.
            actions (Dict[str, Tensor]): The input actions.
            state: Optional[Tensor]: The global state.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The RNN hidden states of critic representation.
            target: If to use target critic network to calculate the critic values.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new RNN hidden states of critic representation.
            values (dict): The evaluated critic values.
        """
        rnn_hidden_new, critic_input = {}, {}

        for key in self.model_keys:
            if self.use_rnn:
                outputs = self.critic_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.critic_representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            critic_input_key = torch.concat([outputs['state'], actions], dim=-1)

            if self.use_global_state:
                critic_input_key = torch.concat([critic_input_key, state], dim=-1)

            if self.use_parameter_sharing:
                critic_input_key = torch.concat([critic_input_key, agent_ids], dim=-1)

            critic_input[key] = critic_input_key

        values = self.target_critic(critic_input) if target else self.critic(critic_input)
        return rnn_hidden_new, values

    def copy_target(self):
        for ep, tp in zip(self.critic_representation.parameters(), self.target_critic_representation.parameters()):
            tp.data.copy_(ep)
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
        self.actor_net = CategoricalActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                             actor_hidden_size, normalize, initialize, kwargs['gain'], activation,
                                             device)
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
