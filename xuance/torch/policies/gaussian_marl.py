import torch
from copy import deepcopy
from typing import Sequence, Optional, Callable, Union, Dict
from gym.spaces import Space, Box, Discrete
from xuance.torch.policies import CriticNet, VDN_mixer
from xuance.torch.utils import ModuleType
from xuance.torch import Tensor, Module
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
        self.actor, self.critic = {}, {}
        for key in self.model_keys:
            dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out = self._get_actor_critic_input(
                self.n_actions[key],
                self.actor_representation[key].output_shapes['state'][0],
                self.critic_representation[key].output_shapes['state'][0], n_agents)

            if self.use_parameter_sharing:
                dim_actor_in += self.n_agents
                dim_critic_in += self.n_agents

            self.actor[key] = GaussianActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                               normalize, initialize, activation, activation_action, device)
            self.critic[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation, device)
        self.mixer = mixer
        self.pi_dist = None

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
            dim_critic_in: The dimension of the input of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in, dim_critic_out = dim_critic_rep, 1
        return dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out

    def forward(self, observation: Tensor, agent_ids: Tensor,
                *rnn_hidden: Tensor, **kwargs):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        actor_input = torch.concat([outputs['state'], agent_ids], dim=-1)
        self.pi_dist = self.actor(actor_input)
        return rnn_hidden, self.pi_dist

    def get_values(self, critic_in: Tensor, agent_ids: Tensor,
                   *rnn_hidden: Tensor, **kwargs):
        shape_input = critic_in.shape
        # get representation features
        if self.use_rnn:
            batch_size, n_agent, episode_length, dim_input = tuple(shape_input)
            outputs = self.representation_critic(critic_in.reshape(-1, episode_length, dim_input), *rnn_hidden)
            outputs['state'] = outputs['state'].reshape(batch_size, n_agent, episode_length, -1)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            batch_size, n_agent, dim_input = tuple(shape_input)
            outputs = self.representation_critic(critic_in.reshape(-1, dim_input))
            outputs['state'] = outputs['state'].reshape(batch_size, n_agent, -1)
            rnn_hidden = None
        # get critic values
        critic_in = torch.concat([outputs['state'], agent_ids], dim=-1)
        v = self.critic(critic_in)
        return rnn_hidden, v

    def value_tot(self, values_n: Tensor, global_state=None):
        if global_state is not None:
            global_state = torch.as_tensor(global_state).to(self.device)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class Basic_ISAC_policy(Module):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(Basic_ISAC_policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.activation_action = activation_action
        self.n_agents = n_agents
        self.representation_info_shape = representation.output_shapes
        dim_input_actor = representation.output_shapes['state'][0]
        dim_input_critic = representation.output_shapes['state'][0] + self.action_dim

        self.actor_representation = representation
        self.actor = GaussianActorNet_SAC(dim_input_actor, n_agents, self.action_dim, actor_hidden_size,
                                          normalize, initialize, activation, activation_action, device)

        self.critic_1_representation = deepcopy(representation)
        self.critic_1 = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.critic_2_representation = deepcopy(representation)
        self.critic_2 = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.target_critic_1_representation = deepcopy(self.critic_1_representation)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2_representation = deepcopy(self.critic_2_representation)
        self.target_critic_2 = deepcopy(self.critic_2)

        self.parameters_actor = list(self.actor_representation.parameters()) + list(self.actor.parameters())
        self.parameters_critic = list(self.critic_1_representation.parameters()) + list(
            self.critic_1.parameters()) + list(self.critic_2_representation.parameters()) + list(
            self.critic_2.parameters())

    def forward(self, observation: Tensor, agent_ids: Tensor):
        outputs_actor = self.actor_representation(observation)
        actor_in = torch.concat([outputs_actor['state'], agent_ids], dim=-1)
        act_dist = self.actor(actor_in)
        act_sample = act_dist.activated_rsample()
        return outputs_actor, act_sample

    def Qpolicy(self, observation: Tensor, agent_ids: Tensor):
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)

        actor_in = torch.concat([outputs_actor['state'], agent_ids], dim=-1)
        act_dist = self.actor(actor_in)
        act_sample, act_log = act_dist.activated_rsample_and_logprob()

        critic_1_in = torch.concat([outputs_critic_1['state'], act_sample, agent_ids], dim=-1)
        critic_2_in = torch.concat([outputs_critic_2['state'], act_sample, agent_ids], dim=-1)
        q_1, q_2 = self.critic_1(critic_1_in), self.critic_2(critic_2_in)
        return act_log, q_1, q_2

    def Qtarget(self, observation: Tensor, agent_ids: Tensor):
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.target_critic_1_representation(observation)
        outputs_critic_2 = self.target_critic_2_representation(observation)

        actor_in = torch.concat([outputs_actor['state'], agent_ids], dim=-1)
        new_act_dist = self.actor(actor_in)
        new_act_sample, new_act_log = new_act_dist.activated_rsample_and_logprob()

        critic_1_in = torch.concat([outputs_critic_1['state'], new_act_sample, agent_ids], dim=-1)
        critic_2_in = torch.concat([outputs_critic_2['state'], new_act_sample, agent_ids], dim=-1)
        target_q_1, target_q_2 = self.target_critic_1(critic_1_in), self.target_critic_2(critic_2_in)
        target_q = torch.min(target_q_1, target_q_2)
        return new_act_log, target_q

    def Qaction(self, observation: Tensor, actions: Tensor, agent_ids: Tensor):
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)
        critic_1_in = torch.concat([outputs_critic_1['state'], actions, agent_ids], dim=-1)
        critic_2_in = torch.concat([outputs_critic_2['state'], actions, agent_ids], dim=-1)
        q_1, q_2 = self.critic_1(critic_1_in), self.critic_2(critic_2_in)
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


class MASAC_policy(Basic_ISAC_policy, Module):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        Module.__init__(self)
        self.action_dim = action_space.shape[0]
        self.activation_action = activation_action
        self.n_agents = n_agents
        self.representation_info_shape = representation.output_shapes
        dim_input_actor = representation.output_shapes['state'][0]
        dim_input_critic = (representation.output_shapes['state'][0] + self.action_dim) * self.n_agents

        self.actor_representation = representation
        self.actor = GaussianActorNet_SAC(dim_input_actor, n_agents, self.action_dim, actor_hidden_size,
                                          normalize, initialize, activation, activation_action, device)

        self.critic_1_representation = deepcopy(representation)
        self.critic_1 = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.critic_2_representation = deepcopy(representation)
        self.critic_2 = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.target_critic_1_representation = deepcopy(self.critic_1_representation)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2_representation = deepcopy(self.critic_2_representation)
        self.target_critic_2 = deepcopy(self.critic_2)

        self.parameters_actor = list(self.actor_representation.parameters()) + list(self.actor.parameters())
        self.parameters_critic = list(self.critic_1_representation.parameters()) + list(
            self.critic_1.parameters()) + list(self.critic_2_representation.parameters()) + list(
            self.critic_2.parameters())

    def Qpolicy(self, observation: Tensor, agent_ids: Tensor):
        bs = observation.shape[0]
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)

        actor_in = torch.concat([outputs_actor['state'], agent_ids], dim=-1)
        act_dist = self.actor(actor_in)
        act_sample, act_log = act_dist.activated_rsample_and_logprob()

        critic_1_in = torch.concat([outputs_critic_1['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    act_sample.view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        critic_2_in = torch.concat([outputs_critic_2['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    act_sample.view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        q_1, q_2 = self.critic_1(critic_1_in), self.critic_2(critic_2_in)
        return act_log, q_1, q_2

    def Qtarget(self, observation: Tensor, agent_ids: Tensor):
        bs = observation.shape[0]
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.target_critic_1_representation(observation)
        outputs_critic_2 = self.target_critic_2_representation(observation)

        actor_in = torch.concat([outputs_actor['state'], agent_ids], dim=-1)
        new_act_dist = self.actor(actor_in)
        new_act_sample, new_act_log = new_act_dist.activated_rsample_and_logprob()

        critic_1_in = torch.concat([outputs_critic_1['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    new_act_sample.view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        critic_2_in = torch.concat([outputs_critic_2['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    new_act_sample.view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        target_q_1, target_q_2 = self.target_critic_1(critic_1_in), self.target_critic_2(critic_2_in)
        target_q = torch.min(target_q_1, target_q_2)
        return new_act_log, target_q

    def Qaction(self, observation: Tensor, actions: Tensor, agent_ids: Tensor):
        bs = observation.shape[0]
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)

        critic_1_in = torch.concat([outputs_critic_1['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    actions.view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        critic_2_in = torch.concat([outputs_critic_2['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    actions.view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        q_1, q_2 = self.critic_1(critic_1_in), self.critic_2(critic_2_in)
        return q_1, q_2
