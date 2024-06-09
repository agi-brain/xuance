import torch
import numpy as np
from copy import deepcopy
from typing import Sequence, Optional, Callable, Union, Dict, List
from gym.spaces import Box
from xuance.torch.policies import CriticNet, VDN_mixer
from xuance.torch.utils import ModuleType
from xuance.torch import Tensor, Module, ModuleDict
from xuance.torch.policies.core import GaussianActorNet, GaussianActorNet_SAC


class MAAC_Policy(Module):
    """
    MAAC_Policy: Multi-Agent Actor-Critic Policy with Gaussian distributions.
    """

    def __init__(self,
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 representation_actor: Module,
                 representation_critic: Module,
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
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

        self.dim_input_critic = {}
        self.actor, self.critic = ModuleDict(), ModuleDict()
        for key in self.model_keys:
            dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out = self._get_actor_critic_input(
                self.action_space[key].shape[-1],
                self.actor_representation[key].output_shapes['state'][0],
                self.critic_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = GaussianActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                               normalize, initialize, activation, activation_action, device)
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
            dim_action: The dimension of actions (continuous), or the number of actions (discrete).
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
        dim_critic_in, dim_critic_out = dim_critic_rep, 1
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

            pi_dists[key] = self.actor[key](actor_input)

        return rnn_hidden, pi_dists

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


class Basic_ISAC_Policy(Module):
    def __init__(self,
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 representation: ModuleDict,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(Basic_ISAC_Policy, self).__init__()
        self.device = device
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.actor_representation = representation
        self.critic_1_representation = deepcopy(representation)
        self.critic_2_representation = deepcopy(representation)
        self.target_critic_1_representation = deepcopy(self.critic_1_representation)
        self.target_critic_2_representation = deepcopy(self.critic_2_representation)

        self.actor, self.critic_1, self.critic_2 = ModuleDict(), ModuleDict(), ModuleDict()
        for key in self.model_keys:
            dim_action = self.action_space[key].shape[-1]
            dim_actor_in, dim_actor_out, dim_critic_in = self._get_actor_critic_input(
                self.actor_representation[key].output_shapes['state'][0], dim_action,
                self.critic_1_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = GaussianActorNet_SAC(dim_actor_in, dim_actor_out, actor_hidden_size,
                                                   normalize, initialize, activation, activation_action, device)
            self.critic_1[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation, device)
            self.critic_2[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation, device)
            self.target_critic_1 = deepcopy(self.critic_1)
            self.target_critic_2 = deepcopy(self.critic_2)

    @property
    def parameters_actor(self):
        parameters_actor = {}
        for key in self.model_keys:
            parameters_actor[key] = list(self.actor_representation[key].parameters()) + list(
                self.actor[key].parameters())
        return parameters_actor

    @property
    def parameters_critic(self):
        parameters_critic = {}
        for key in self.model_keys:
            parameters_critic[key] = list(self.critic_1_representation[key].parameters()) + list(
                self.critic_1[key].parameters()) + list(self.critic_2_representation[key].parameters()) + list(
                self.critic_2[key].parameters())
        return parameters_critic

    def _get_actor_critic_input(self, dim_actor_rep, dim_action, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor netwrok and critic networks.

        Parameters:
            dim_actor_rep: The dimension of the output of actor presentation.
            dim_action: The dimension of actions (continuous), or the number of actions (discrete).
            dim_critic_rep: The dimension of the output of critic presentation.
            n_agents: The number of agents.

        Returns:
            dim_actor_in: The dimension of input of the actor networks.
            dim_actor_out: The dimension of output of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
            dim_critic_out: The dimension of the output of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in = dim_critic_rep + dim_action
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in

    def forward(self, observation: Dict[str, Tensor],
                agent_ids: Tensor = None, agent_key: str = None):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.

        Returns:
            actions (Dict[Tensor]): The actions output by the policies.
        """
        act_dists, actions_dict = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            outputs = self.actor_representation[key](observation[key])
            if self.use_parameter_sharing:
                actor_in = torch.concat([outputs['state'], agent_ids], dim=-1)
            else:
                actor_in = outputs['state']
            act_dists = self.actor[key](actor_in)
            actions_dict[key] = act_dists.activated_rsample()
        return outputs, actions_dict

    def Qpolicy(self, observation: Dict[str, Tensor], agent_ids: Tensor = None, agent_key: str = None):
        """
        Returns Q^policy of current observations and actions pairs.

        Parameters:
            observation (Dict[Tensor]): The observations.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.

        Returns:
            q_eval: The evaluations of Q^policy.
        """
        log_action_prob, q_1, q_2 = {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            outputs_actor = self.actor_representation[key](observation[key])
            outputs_critic_1 = self.critic_1_representation[key](observation[key])
            outputs_critic_2 = self.critic_2_representation[key](observation[key])

            actor_in = outputs_actor['state']
            if self.use_parameter_sharing:
                actor_in = torch.concat([outputs_actor['state'], agent_ids], dim=-1)
            act_dist = self.actor[key](actor_in)
            act_sample, log_action_prob[key] = act_dist.activated_rsample_and_logprob()

            critic_1_in = torch.concat([outputs_critic_1['state'], act_sample], dim=-1)
            critic_2_in = torch.concat([outputs_critic_2['state'], act_sample], dim=-1)
            if self.use_parameter_sharing:
                critic_1_in = torch.concat([critic_1_in, agent_ids], dim=-1)
                critic_2_in = torch.concat([critic_2_in, agent_ids], dim=-1)
            q_1[key], q_2[key] = self.critic_1[key](critic_1_in), self.critic_2[key](critic_2_in)
        return log_action_prob, q_1, q_2

    def Qtarget(self, next_observation: Dict[str, Tensor], agent_ids: Tensor = None, agent_key: str = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            next_observation (Dict[Tensor]): The observations of next step.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.

        Returns:
            q_target: The evaluations of Q^target.
        """
        new_act_log, target_q = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            outputs_actor = self.actor_representation[key](next_observation[key])
            outputs_critic_1 = self.target_critic_1_representation[key](next_observation[key])
            outputs_critic_2 = self.target_critic_2_representation[key](next_observation[key])

            actor_in = outputs_actor['state']
            if self.use_parameter_sharing:
                actor_in = torch.concat([actor_in, agent_ids], dim=-1)
            new_act_dist = self.actor[key](actor_in)
            new_act_sample, new_act_log[key] = new_act_dist.activated_rsample_and_logprob()

            critic_1_in = torch.concat([outputs_critic_1['state'], new_act_sample], dim=-1)
            critic_2_in = torch.concat([outputs_critic_2['state'], new_act_sample], dim=-1)
            if self.use_parameter_sharing:
                critic_1_in = torch.concat([critic_1_in, agent_ids], dim=-1)
                critic_2_in = torch.concat([critic_2_in, agent_ids], dim=-1)
            target_q_1, target_q_2 = self.target_critic_1[key](critic_1_in), self.target_critic_2[key](critic_2_in)
            target_q[key] = torch.min(target_q_1, target_q_2)
        return new_act_log, target_q

    def Qaction(self, observation: Union[np.ndarray, dict], actions: Tensor,
                agent_ids: Tensor, agent_key: str = None):
        """
        Returns the evaluated Q-values for current observation-action pairs.

        Parameters:
            observation: The original observation.
            actions: The selected actions.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.

        Returns:
            q_1: The Q-value calculated by the first critic network.
            q_2: The Q-value calculated by the other critic network.
        """
        q_1, q_2 = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            outputs_critic_1 = self.critic_1_representation[key](observation[key])
            outputs_critic_2 = self.critic_2_representation[key](observation[key])

            critic_1_in = torch.concat([outputs_critic_1['state'], actions[key]], dim=-1)
            critic_2_in = torch.concat([outputs_critic_2['state'], actions[key]], dim=-1)
            if self.use_parameter_sharing:
                critic_1_in = torch.concat([critic_1_in, agent_ids], dim=-1)
                critic_2_in = torch.concat([critic_2_in, agent_ids], dim=-1)
            q_1[key], q_2[key] = self.critic_1[key](critic_1_in), self.critic_2[key](critic_2_in)
        return q_1, q_2

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic_1_representation.parameters(), self.target_critic_1_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2_representation.parameters(), self.target_critic_2_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class MASAC_Policy(Basic_ISAC_Policy):
    def __init__(self,
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 representation: ModuleDict,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(MASAC_Policy, self).__init__(action_space, n_agents, representation,
                                           actor_hidden_size, critic_hidden_size,
                                           normalize, initialize, activation, activation_action, device, **kwargs)

    def _get_actor_critic_input(self, dim_actor_rep, dim_action, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor netwrok and critic networks.

        Parameters:
            dim_actor_rep: The dimension of the output of actor presentation.
            dim_action: The dimension of actions (continuous), or the number of actions (discrete).
            dim_critic_rep: The dimension of the output of critic presentation.
            n_agents: The number of agents.

        Returns:
            dim_actor_in: The dimension of input of the actor networks.
            dim_actor_out: The dimension of output of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
            dim_critic_out: The dimension of the output of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in = (dim_critic_rep + dim_action) * n_agents
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in

    def Qpolicy(self, observation: Dict[str, Tensor], agent_ids: Tensor = None, agent_key: str = None):
        """
        Returns Q^policy of current observations and actions pairs.

        Parameters:
            observation (Dict[Tensor]): The observations.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.

        Returns:
            q_eval: The evaluations of Q^policy.
        """
        bs = observation[self.model_keys[0]].shape[0]
        act_sample, outputs_critic_1, outputs_critic_2 = {}, {}, {}
        log_action_prob, q_1, q_2 = {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in self.model_keys:
            outputs_actor = self.actor_representation[key](observation[key])
            outputs_critic_1[key] = self.critic_1_representation[key](observation[key])['state']
            outputs_critic_2[key] = self.critic_2_representation[key](observation[key])['state']

            actor_in = outputs_actor['state']
            if self.use_parameter_sharing:
                actor_in = torch.concat([outputs_actor['state'], agent_ids], dim=-1)
            act_dist = self.actor[key](actor_in)
            act_sample[key], log_action_prob[key] = act_dist.activated_rsample_and_logprob()

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            joint_act_in = act_sample[key].reshape(bs, -1).unsqueeze(1).expand(-1, self.n_agents, -1)
            joint_obs_in_1 = outputs_critic_1[key].reshape(bs, -1).unsqueeze(1).expand(-1, self.n_agents, -1)
            joint_obs_in_2 = outputs_critic_2[key].reshape(bs, -1).unsqueeze(1).expand(-1, self.n_agents, -1)
            critic_1_in = torch.concat([joint_obs_in_1, joint_act_in, agent_ids], dim=-1)
            critic_2_in = torch.concat([joint_obs_in_2, joint_act_in, agent_ids], dim=-1)
            q_1[key], q_2[key] = self.critic_1[key](critic_1_in), self.critic_2[key](critic_2_in)
        else:
            joint_act_in = torch.concat([act_sample[k].unsqueeze(1) for k in self.model_keys], dim=1).reshape(bs, -1)
            joint_obs_in_1 = torch.concat([outputs_critic_1[k].reshape(bs, 1, -1) for k in self.model_keys], dim=-1)
            joint_obs_in_1 = joint_obs_in_1.reshape(bs, -1)
            joint_obs_in_2 = torch.concat([outputs_critic_2[k].reshape(bs, 1, -1) for k in self.model_keys], dim=-1)
            joint_obs_in_2 = joint_obs_in_2.reshape(bs, -1)
            joint_critic_in_1 = torch.concat([joint_obs_in_1, joint_act_in], dim=-1)
            joint_critic_in_2 = torch.concat([joint_obs_in_2, joint_act_in], dim=-1)

            for key in agent_list:
                q_1[key], q_2[key] = self.critic_1[key](joint_critic_in_1), self.critic_2[key](joint_critic_in_2)

        return log_action_prob, q_1, q_2

    def Qtarget(self, next_observation: Dict[str, Tensor], agent_ids: Tensor = None, agent_key: str = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            next_observation (Dict[Tensor]): The observations of next step.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.

        Returns:
            q_target: The evaluations of Q^target.
        """
        bs = next_observation[self.model_keys[0]].shape[0]
        act_sample, outputs_critic_1, outputs_critic_2 = {}, {}, {}
        new_act_log, target_q = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in self.model_keys:
            outputs_actor = self.actor_representation[key](next_observation[key])
            outputs_critic_1[key] = self.critic_1_representation[key](next_observation[key])['state']
            outputs_critic_2[key] = self.critic_2_representation[key](next_observation[key])['state']

            actor_in = outputs_actor['state']
            if self.use_parameter_sharing:
                actor_in = torch.concat([outputs_actor['state'], agent_ids], dim=-1)
            act_dist = self.actor[key](actor_in)
            act_sample[key], new_act_log[key] = act_dist.activated_rsample_and_logprob()

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            joint_act_in = act_sample[key].reshape(bs, -1).unsqueeze(1).expand(-1, self.n_agents, -1)
            joint_obs_in_1 = outputs_critic_1[key].reshape(bs, -1).unsqueeze(1).expand(-1, self.n_agents, -1)
            joint_obs_in_2 = outputs_critic_2[key].reshape(bs, -1).unsqueeze(1).expand(-1, self.n_agents, -1)
            critic_1_in = torch.concat([joint_obs_in_1, joint_act_in, agent_ids], dim=-1)
            critic_2_in = torch.concat([joint_obs_in_2, joint_act_in, agent_ids], dim=-1)
            q_1, q_2 = self.critic_1[key](critic_1_in), self.critic_2[key](critic_2_in)
            target_q[key] = torch.min(q_1, q_2)
        else:
            joint_act_in = torch.concat([act_sample[k].unsqueeze(1) for k in self.model_keys], dim=1).reshape(bs, -1)
            joint_obs_in_1 = torch.concat([outputs_critic_1[k].reshape(bs, 1, -1) for k in self.model_keys], dim=-1)
            joint_obs_in_1 = joint_obs_in_1.reshape(bs, -1)
            joint_obs_in_2 = torch.concat([outputs_critic_2[k].reshape(bs, 1, -1) for k in self.model_keys], dim=-1)
            joint_obs_in_2 = joint_obs_in_2.reshape(bs, -1)
            joint_critic_in_1 = torch.concat([joint_obs_in_1, joint_act_in], dim=-1)
            joint_critic_in_2 = torch.concat([joint_obs_in_2, joint_act_in], dim=-1)

            for key in agent_list:
                q_1, q_2 = self.critic_1[key](joint_critic_in_1), self.critic_2[key](joint_critic_in_2)
                target_q[key] = torch.min(q_1, q_2)

        return new_act_log, target_q

    def Qaction(self, observation: Union[np.ndarray, dict], actions: Tensor,
                agent_ids: Tensor, agent_key: str = None):
        """
        Returns the evaluated Q-values for current observation-action pairs.

        Parameters:
            observation: The original observation.
            actions: The selected actions.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.

        Returns:
            q_1: The Q-value calculated by the first critic network.
            q_2: The Q-value calculated by the other critic network.
        """
        bs = observation[self.model_keys[0]].shape[0]
        outputs_critic_1, outputs_critic_2 = {}, {}
        q_1, q_2 = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in self.model_keys:
            outputs_critic_1[key] = self.critic_1_representation[key](observation[key])['state']
            outputs_critic_2[key] = self.critic_2_representation[key](observation[key])['state']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            joint_act_in = actions[key].reshape(bs, 1, -1).expand(-1, self.n_agents, -1)
            joint_obs_in_1 = outputs_critic_1[key].reshape(bs, -1).unsqueeze(1).expand(-1, self.n_agents, -1)
            joint_obs_in_2 = outputs_critic_2[key].reshape(bs, -1).unsqueeze(1).expand(-1, self.n_agents, -1)
            critic_1_in = torch.concat([joint_obs_in_1, joint_act_in, agent_ids], dim=-1)
            critic_2_in = torch.concat([joint_obs_in_2, joint_act_in, agent_ids], dim=-1)
            q_1[key], q_2[key] = self.critic_1[key](critic_1_in), self.critic_2[key](critic_2_in)
        else:
            joint_act_in = torch.concat([actions[k].unsqueeze(1) for k in self.model_keys], dim=1).reshape(bs, -1)
            joint_obs_in_1 = torch.concat([outputs_critic_1[k].reshape(bs, 1, -1) for k in self.model_keys], dim=-1)
            joint_obs_in_1 = joint_obs_in_1.reshape(bs, -1)
            joint_obs_in_2 = torch.concat([outputs_critic_2[k].reshape(bs, 1, -1) for k in self.model_keys], dim=-1)
            joint_obs_in_2 = joint_obs_in_2.reshape(bs, -1)
            joint_critic_in_1 = torch.concat([joint_obs_in_1, joint_act_in], dim=-1)
            joint_critic_in_2 = torch.concat([joint_obs_in_2, joint_act_in], dim=-1)

            for key in agent_list:
                q_1[key], q_2[key] = self.critic_1[key](joint_critic_in_1), self.critic_2[key](joint_critic_in_2)
        return q_1, q_2
