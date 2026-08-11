import torch
from copy import deepcopy
from typing import Type, Sequence, Optional, Callable, Union, Dict, Any
from gymnasium.spaces import Discrete, Box
from torch import Tensor
from torch.nn import Module, ModuleDict
from xuance.common import AgentGrouping
from xuance.torch.rl_models.heads import ValueHead, QValueHead
from xuance.torch.rl_models.modules import MultiAgentModelOutput, CriticOutput, TwinCriticOutput, RNN_State


class CentralizedStateValueCritic(Module):
    """V(s, {o^1, ..., o^N}). Typically, for MAPPO-like algorithms"""

    def __init__(self,
                 grouping: AgentGrouping,
                 representations: ModuleDict,
                 state_space: Box,
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 use_rnn: bool = False,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__()

        self.grouping = grouping
        self.groups = grouping.groups
        self.group_keys = grouping.group_keys
        self.agent_keys = grouping.agent_keys
        self.n_agents = len(self.agent_keys)
        self.n_group_agents = {k: len(self.groups[k]) for k in self.group_keys}
        self.use_rnn = use_rnn
        self.device = device

        self.state_space = state_space
        if isinstance(state_space, Box):
            self.state_dim = state_space.shape[0]
        elif state_space is None:
            self.state_dim = 0
        else:
            raise NotImplementedError
        self.representations = representations
        self.representation_info_shape = {k: representations[k].output_shapes for k in self.group_keys}
        self.sum_obs_embedding_dim = sum([self.representation_info_shape[k]['state'][0] for k in self.group_keys])

        self.critic_head = ValueHead(
            feature_dim=self.state_dim + self.sum_obs_embedding_dim,
            hidden_size=critic_hidden_size,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device,
            **kwargs,
        )

    def forward(self,
                observations: Dict[str, Tensor],
                agent_indices: Dict[str, Tensor],
                expanded_states: Optional[Tensor] = None,  # shape: batch * T * dim_S or batch * T * N * dim_S (for RNN)
                rnn_states: Dict[str, RNN_State | dict] = None,
                **kwargs) -> MultiAgentModelOutput:
        rnn_states_new, rep_out, obs_features, evalQ = {}, {}, {}, {}
        bs = observations[self.group_keys[0]].shape[0]
        seq_len = observations[self.group_keys[0]].shape[1] if self.use_rnn else 1

        for group, group_agents in self.groups.items():
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent

            if self.use_rnn:
                representation_output = self.representations[group](observations[group],
                                                                    rnn_states=rnn_states[group],
                                                                    agent_indices=agent_indices[group])
                # Features shape: batch_size * n_agent * seq_len * feature_dim
                group_obs_features = representation_output.embeddings.reshape(batch_size, n_agent, seq_len, -1)
            else:
                representation_output = self.representations[group](observations[group],
                                                                    agent_indices=agent_indices[group])
                # Features shape: batch_size * n_agent * feature_dim
                group_obs_features = representation_output.embeddings.reshape(batch_size, n_agent, -1)

            rep_out[group] = representation_output
            rnn_states_new[group] = representation_output.rnn_states
            for i, agent_key in enumerate(group_agents):
                obs_features[agent_key] = group_obs_features[:, i]

        critic_input = torch.stack([obs_features[k] for k in self.agent_keys], dim=1)  # batch * N * ...
        if expanded_states is not None:
            critic_input = torch.concat([expanded_states, critic_input], dim=-1)
        values = self.critic_head(critic_input, **kwargs)

        return MultiAgentModelOutput(values=values, critic_rnn_states=rnn_states_new, critic_rep_out=rep_out)


class CentralizedActionValueCritic(Module):
    """Q(o^1, ..., o^N, a^1, ..., a^N). Typically, for MADDPG-like algorithm"""

    def __init__(self,
                 representation: Module,
                 action_space: Dict[str, Box],
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__()

        self.action_space = action_space
        self.action_dim = {k: v.shape[0] for k, v in action_space.items()}
        self.joint_action_dim = sum(self.action_dim.values())
        self.representation = representation
        self.representation_info_shape = representation.output_shapes

        self.critic_head = ValueHead(
            feature_dim=self.representation_info_shape['state'][0] + self.joint_action_dim,
            hidden_size=critic_hidden_size,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device,
            **kwargs,
        )

    def forward(self,
                joint_observations: Union[Tensor, dict],
                joint_actions: Union[Tensor, dict],
                **kwargs) -> CriticOutput:
        joint_obs_rep_out = self.representation(joint_observations, **kwargs)
        return CriticOutput(
            representations=joint_obs_rep_out,
            values=self.critic_head(torch.concat([joint_obs_rep_out.embeddings, joint_actions], dim=-1), **kwargs)
        )


class TwinCentralizedActionValueCritic(Module):
    """
    Q^1(o^1, ..., o^N, a^1, ..., a^N), Q^2(o^1, ..., o^N, a^1, ..., a^N).
    Typically, for MASAC-like algorithm
    """

    def __init__(self,
                 representation: Module,
                 action_space: Dict[str, Box] | Any,
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__()

        self.action_space = action_space
        self.action_dim = {k: v.shape[0] for k, v in action_space.items()}
        self.joint_action_dim = sum(self.action_dim.values())
        self.representation_1 = representation
        self.representation_2 = deepcopy(representation)
        self.representation_info_shape = representation.output_shapes

        self.feature_dim = self.representation_info_shape['state'][0] + self.joint_action_dim

        value_head_input = dict(
            feature_dim=self.feature_dim,
            hidden_size=critic_hidden_size,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device,
            **kwargs,
        )

        self.critic_head_1 = ValueHead(**value_head_input)
        self.critic_head_2 = ValueHead(**value_head_input)

    def forward(self,
                joint_observations: Union[Tensor, dict],
                joint_actions: Union[Tensor, dict],
                rnn_states_1: Dict[str, RNN_State | dict] = None,
                rnn_states_2: Dict[str, RNN_State | dict] = None,
                **kwargs) -> TwinCriticOutput:
        if rnn_states_1 is not None:
            kwargs["rnn_states"] = rnn_states_1
        rep_out_1 = self.representation_1(joint_observations, **kwargs)
        if rnn_states_1 is not None:
            kwargs["rnn_states"] = rnn_states_2
        rep_out_2 = self.representation_2(joint_observations, **kwargs)
        return TwinCriticOutput(
            representations_1=rep_out_1,
            representations_2=rep_out_2,
            values_1=self.critic_head_1(torch.concat([rep_out_1.embeddings, joint_actions], dim=-1), **kwargs),
            values_2=self.critic_head_2(torch.concat([rep_out_2.embeddings, joint_actions], dim=-1), **kwargs)
        )


class CounterfactualCentralizedCritic(Module):
    """Q(s, o^i, {a^1, ..., a^N}\a^i). Typically, for COMA's centralized critic,"""

    def __init__(self,
                 grouping: AgentGrouping,
                 representations: ModuleDict,
                 state_space: Box,
                 action_space: Dict[str, Discrete],
                 critic_hidden_size: Sequence[int],
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 use_rnn: bool = False,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__()

        self.grouping = grouping
        self.groups = grouping.groups
        self.group_keys = grouping.group_keys
        self.agent_keys = grouping.agent_keys
        self.n_agents = len(self.agent_keys)
        self.n_group_agents = {k: len(self.groups[k]) for k in self.group_keys}
        self.use_rnn = use_rnn
        self.device = device

        self.state_space = state_space
        self.action_space = action_space
        self.state_dim = state_space.shape[0]
        self.n_actions = {k: v.n for k, v in action_space.items()}
        self.joint_action_dim = sum(self.n_actions.values())
        self.representations = representations
        self.representation_info_shape = representations[self.group_keys[0]].output_shapes

        self.critic_head = QValueHead(
            feature_dim=self.state_dim + self.representation_info_shape['state'][0] + self.joint_action_dim,
            n_actions=max(self.n_actions.values()),
            hidden_size=critic_hidden_size,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device,
            **kwargs,
        )

    def forward(self,
                states: Tensor,
                observations: Dict[str, Tensor],
                joint_actions: Tensor,
                agent_indices: Dict[str, Tensor],
                rnn_states: Dict[str, RNN_State | dict] = None,
                **kwargs) -> MultiAgentModelOutput:
        rnn_states_new, rep_out, evalQ = {}, {}, {}
        bs = observations[self.group_keys[0]].shape[0]

        if self.use_rnn:
            seq_len = states.shape[1]
            expanded_states = states.unsqueeze(1).repeat(1, self.n_agents, 1, 1)  # batch * T * N * dim_S
            expanded_joint_actions = joint_actions.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
        else:
            seq_len = 1
            expanded_states = states.unsqueeze(1).repeat(1, self.n_agents, 1)  # batch * N * dim_S
            expanded_joint_actions = joint_actions.unsqueeze(1).repeat(1, self.n_agents, 1)

        agent_mask = (1 - torch.eye(self.n_agents, dtype=torch.float32, device=self.device)).unsqueeze(-1)
        for group, group_agents in self.groups.items():
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent

            agent_mask = agent_mask.repeat(1, 1, self.critic_head.n_actions).reshape(n_agent, -1).unsqueeze(0)

            if self.use_rnn:
                agent_mask = agent_mask.unsqueeze(2)
                representation_output = self.representations[group](observations[group],
                                                                    rnn_states=rnn_states[group],
                                                                    agent_indices=agent_indices[group])
                # Features shape: batch_size * n_agent * seq_len * feature_dim
                group_obs_features = representation_output.embeddings.reshape(batch_size, n_agent, seq_len, -1)
            else:
                representation_output = self.representations[group](observations[group],
                                                                    agent_indices=agent_indices[group])
                # Features shape: batch_size * n_agent * feature_dim
                group_obs_features = representation_output.embeddings.reshape(batch_size, n_agent, -1)

            rep_out[group] = representation_output
            rnn_states_new[group] = representation_output.rnn_states
            masked_joint_actions = expanded_joint_actions * agent_mask
            critic_input = torch.concat([expanded_states, group_obs_features, masked_joint_actions], dim=-1)
            critic_output = self.critic_head(critic_input, **kwargs)
            for i, agent_key in enumerate(group_agents):
                evalQ[agent_key] = critic_output[:, i]

        return MultiAgentModelOutput(values=evalQ, critic_rnn_states=rnn_states_new, critic_rep_out=rep_out)
