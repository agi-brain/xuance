import os
import torch
from copy import deepcopy
from typing import Dict, Optional, Union, Tuple
from torch import Tensor
from torch.nn import Module, ModuleDict
from torch.nn.parallel import DistributedDataParallel
from xuance.common import AgentGrouping
from xuance.torch.rl_models.modules import RNN_State, MultiAgentModelOutput
from .base import OffPolicyMultiAgentActorCritic


class IndependentActorCritic(Module):
    def __init__(self,
                 grouping: AgentGrouping,
                 actors: ModuleDict,
                 critics: Module | ModuleDict,
                 use_rnn: bool = False,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
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

        self.actors = actors
        self.critics = critics

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            self.actors = DistributedDataParallel(module=self.actors, device_ids=[self.rank])
            self.critics = DistributedDataParallel(module=self.critics, device_ids=[self.rank])

    @property
    def parameters_model(self):
        return {
            group: list(self.actors[group].parameters()) + list(self.critics[group].parameters())
            for group in self.group_keys
        }

    def forward(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            avail_actions: Dict[str, Tensor] = None,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            deterministic: bool = False
    ) -> MultiAgentModelOutput:
        rnn_states_new, pi_dists, actions = {}, {}, {}
        input_shape = observations[self.group_keys[0]].shape
        bs = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            group_agents = self.groups[group]
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            actor_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if avail_actions is not None:
                actor_kwargs["avail_actions"] = avail_actions[group]
            if self.use_rnn:
                actor_kwargs["rnn_states"] = rnn_states[group]

            actor_out = self.actors[group](observations[group], **actor_kwargs)

            policy_dist = actor_out.distributions
            if deterministic:
                sampled_actions = policy_dist.deterministic_sample()
            else:
                sampled_actions = policy_dist.stochastic_sample()
            group_actions = sampled_actions.reshape(*batch_shape, -1)

            rnn_states_new[group] = actor_out.representations.rnn_states
            pi_dists[group] = actor_out.distributions

            for i, agent_key in enumerate(group_agents):
                actions[agent_key] = group_actions[:, i]

        return MultiAgentModelOutput(actions=actions, distributions=pi_dists, actor_rnn_states=rnn_states_new)

    def get_values(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            states: Tensor = None,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> MultiAgentModelOutput:
        rnn_states_new, values = {}, {}
        input_shape = observations[self.group_keys[0]].shape
        bs = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            group_agents = self.groups[group]
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            critic_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                critic_kwargs["rnn_states"] = rnn_states[group]

            critic_out = self.critics[group](observations[group], **critic_kwargs)

            group_values = critic_out.values.reshape(*batch_shape, 1)

            rnn_states_new[group] = critic_out.representations.rnn_states

            for i, agent_key in enumerate(group_agents):
                values[agent_key] = group_values[:, i]

        return MultiAgentModelOutput(values=values, critic_rnn_states=rnn_states_new)

    def init_actor_rnn_states(self, batch_size: int) -> Dict[str, RNN_State] | None:
        rnn_states = None
        if self.use_rnn:
            rnn_states = {}
            for group in self.group_keys:
                bs = batch_size * self.n_group_agents[group]
                rnn_states[group] = self.actors[group].representation.obs_representation.init_rnn_states(bs)
        return rnn_states

    def init_actor_rnn_states_item(self, i_env: int,
                                   rnn_states: Dict[str, RNN_State] = None) -> Dict[str, RNN_State]:
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        batch_index = [i_env, ]
        for group in self.group_keys:
            rnn_states[group] = self.actors[group].representation.obs_representation.init_rnn_states_item(
                batch_index, rnn_states[group])
        return rnn_states

    def init_critic_rnn_states(self, batch_size: int) -> Dict[str, RNN_State] | None:
        rnn_states = None
        if self.use_rnn:
            rnn_states = {}
            for group in self.group_keys:
                bs = batch_size * self.n_group_agents[group]
                rnn_states[group] = self.critics[group].representation.obs_representation.init_rnn_states(bs)
        return rnn_states

    def init_critic_rnn_states_item(self, i_env: int,
                                    rnn_states: Dict[str, RNN_State] = None) -> Dict[str, RNN_State]:
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        batch_index = [i_env, ]
        for group in self.group_keys:
            rnn_states[group] = self.critics[group].representation.obs_representation.init_rnn_states_item(
                batch_index, rnn_states[group])
        return rnn_states


class MultiAgentActorCritic(IndependentActorCritic):
    def get_values(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            states: Tensor = None,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> MultiAgentModelOutput:
        rnn_states_new, values = {}, {}
        input_shape = observations[self.group_keys[0]].shape
        bs = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        if states is not None:
            if self.use_rnn:
                expanded_states = states.unsqueeze(1).repeat(1, self.n_agents, 1, 1)  # batch * T * N * dim_S
            else:
                expanded_states = states.unsqueeze(1).repeat(1, self.n_agents, 1)  # batch * N * dim_S
        else:
            expanded_states = None

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            group_agents = self.groups[group]
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            critic_kwargs = {
                "expanded_states": expanded_states,
                "observations": observations,
                "agent_indices": agent_indices
            }
            if self.use_rnn:
                critic_kwargs["rnn_states"] = rnn_states[group]

            critic_out = self.critics[group](**critic_kwargs)

            group_values = critic_out.values.reshape(*batch_shape, 1)

            rnn_states_new[group] = critic_out.critic_rnn_states

            for i, agent_key in enumerate(group_agents):
                values[agent_key] = group_values[:, i]

        return MultiAgentModelOutput(values=values, critic_rnn_states=rnn_states_new)

    def init_critic_rnn_states(self, batch_size: int) -> Dict[str, Dict[str, RNN_State]] | None:
        rnn_states = None
        if self.use_rnn:
            rnn_states = {}
            for group in self.group_keys:
                bs = batch_size * self.n_group_agents[group]
                rnn_states[group] = {k: self.critics[group].representations[k].obs_representation.init_rnn_states(bs)
                                     for k in self.group_keys}
        return rnn_states

    def init_critic_rnn_states_item(
            self,
            i_env: int,
            rnn_states: Dict[str, Dict[str, RNN_State]] = None
    ) -> Dict[str, Dict[str, RNN_State]]:
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        batch_index = [i_env, ]
        for group in self.group_keys:
            rnn_states[group] = {k: self.critics[group].representations[k].obs_representation.init_rnn_states_item(
                batch_index, rnn_states[group][k]) for k in self.group_keys}
        return rnn_states


class CounterfactualMultiAgentActorCritic(IndependentActorCritic):
    def __init__(self, *args, **kwargs) -> None:
        super(CounterfactualMultiAgentActorCritic, self).__init__(*args, **kwargs)
        self.target_critics = deepcopy(self.critics)

    def forward(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            avail_actions: Dict[str, Tensor] = None,
            epsilon: float = 0.0,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            deterministic: bool = False,
            test_mode: bool = False
    ) -> MultiAgentModelOutput:
        rnn_states_new, pi_dists, actions = {}, {}, {}
        input_shape = observations[self.group_keys[0]].shape
        bs = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            group_agents = self.groups[group]
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            actor_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if avail_actions is not None:
                actor_kwargs["avail_actions"] = avail_actions[group]
            if self.use_rnn:
                actor_kwargs["rnn_states"] = rnn_states[group]

            actor_out = self.actors[group](observations[group], **actor_kwargs)

            group_probs = actor_out.distributions.probs

            if not test_mode:
                group_probs = (1 - epsilon) * group_probs + epsilon * 1 / self.actors[group].action_dim

            self.actors[group].actor_head.policy_distribution.set_param(probs=group_probs)
            policy_dist = self.actors[group].actor_head.policy_distribution

            if deterministic:
                sampled_actions = policy_dist.deterministic_sample()
            else:
                sampled_actions = policy_dist.stochastic_sample()

            group_actions = sampled_actions.reshape(*batch_shape, -1)

            rnn_states_new[group] = actor_out.representations.rnn_states
            pi_dists[group] = policy_dist

            for i, agent_key in enumerate(group_agents):
                actions[agent_key] = group_actions[:, i]

        return MultiAgentModelOutput(actions=actions, distributions=pi_dists, actor_rnn_states=rnn_states_new)

    def get_values(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            states: Tensor = None,
            joint_actions: Tensor = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            target: bool = False,
            **kwargs
    ) -> MultiAgentModelOutput:
        critic = self.target_critics if target else self.critics
        return critic(
            states=states,
            observations=observations,
            joint_actions=joint_actions,
            agent_indices=agent_indices,
            rnn_states=rnn_states,
            **kwargs
        )

    def init_critic_rnn_states(self, batch_size: int) -> Dict[str, RNN_State] | None:
        rnn_states = None
        if self.use_rnn:
            rnn_states = {}
            for group in self.group_keys:
                bs = batch_size * self.n_group_agents[group]
                rnn_states[group] = self.critics.representations[group].obs_representation.init_rnn_states(bs)
        return rnn_states

    def init_critic_rnn_states_item(self, i_env: int,
                                    rnn_states: Dict[str, RNN_State] = None) -> Dict[str, RNN_State]:
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        batch_index = [i_env, ]
        for group in self.group_keys:
            rnn_states[group] = self.critics.representations[group].obs_representation.init_rnn_states_item(
                batch_index, rnn_states[group])
        return rnn_states

    def copy_target(self):
        for ep, tp in zip(self.critics.parameters(), self.target_critics.parameters()):
            tp.data.copy_(ep)


class ValueDecompositionActorCritic(IndependentActorCritic):
    def __init__(self,
                 grouping: AgentGrouping,
                 actors: ModuleDict,
                 critics: Module | ModuleDict,
                 mixer: Module,
                 use_rnn: bool = False,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__(
            grouping=grouping,
            actors=actors,
            critics=critics,
            use_rnn=use_rnn,
            device=device,
            use_distributed_training=use_distributed_training,
            **kwargs
        )
        self.v_tot = mixer
        # Prepare DDP module.
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            self.v_tot = DistributedDataParallel(module=self.v_tot, device_ids=[self.rank])

    def values_tot(self, individual_values: Dict[str, Tensor], global_states: Optional[Tensor] = None):
        # Expected shape: [tot_batch_size * 1, ...] -> tot_batch_size * n_agents_all
        individual_inputs = torch.concat([individual_values[k].reshape([-1, 1]) for k in self.agent_keys], dim=-1)
        # Output shape: tot_batch_size * 1
        values = self.v_tot(individual_inputs, global_states)
        return values


class IndependentDeterministicActorCritic(OffPolicyMultiAgentActorCritic):
    def forward(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> MultiAgentModelOutput:
        rnn_states_new, actions = {}, {}
        input_shape = observations[self.group_keys[0]].shape
        bs = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            group_agents = self.groups[group]
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            actor_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                actor_kwargs["rnn_states"] = rnn_states[group]

            actor_out = self.actors[group](observations[group], **actor_kwargs)

            group_actions = actor_out.actions.reshape(*batch_shape, -1)

            rnn_states_new[group] = actor_out.representations.rnn_states

            for i, agent_key in enumerate(group_agents):
                actions[agent_key] = group_actions[:, i]

        return MultiAgentModelOutput(actions=actions, actor_rnn_states=rnn_states_new)

    def Qpolicy(
            self,
            observations: Dict[str, Tensor],
            actions: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Dict[str, Tensor]:
        q_eval = {}

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:

            critic_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                critic_kwargs["rnn_states"] = rnn_states[group]

            critic_out = self.critics[group](observations[group], actions[group], **critic_kwargs)

            q_eval[group] = critic_out.values

        return q_eval

    def Qtarget(
            self,
            observations: Dict[str, Tensor],
            actions: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Dict[str, Tensor]:
        q_target = {}

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:

            critic_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                critic_kwargs["rnn_states"] = rnn_states[group]

            target_critic_out = self.target_critics[group](observations[group], actions[group], **critic_kwargs)

            q_target[group] = target_critic_out.values

        return q_target

    def Atarget(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Dict[str, Tensor]:
        actions = {}
        input_shape = observations[self.group_keys[0]].shape
        bs = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            group_agents = self.groups[group]
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            actor_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                actor_kwargs["rnn_states"] = rnn_states[group]

            target_actor_out = self.target_actors[group](observations[group], **actor_kwargs)

            group_actions = target_actor_out.actions.reshape(*batch_shape, -1)

            for i, agent_key in enumerate(group_agents):
                actions[agent_key] = group_actions[:, i]

        return actions


class MultiAgentDeterministicActorCritic(IndependentDeterministicActorCritic):
    def Qpolicy(
            self,
            joint_observations: Tensor,
            joint_actions: Tensor,
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Dict[str, Tensor]:
        q_eval = {}
        input_shape = joint_observations.shape
        bs = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            group_agents = self.groups[group]
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            if self.use_rnn:
                expand_joint_obs = joint_observations.unsqueeze(1).expand(-1, n_agent, -1, -1).reshape(bs, seq_len, -1)
                expand_joint_act = joint_actions.unsqueeze(1).expand(-1, n_agent, -1, -1).reshape(bs, seq_len, -1)
            else:
                expand_joint_obs = joint_observations.unsqueeze(1).expand(-1, n_agent, -1).reshape(bs, -1)
                expand_joint_act = joint_actions.unsqueeze(1).expand(-1, n_agent, -1).reshape(bs, -1)

            critic_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                critic_kwargs["rnn_states"] = rnn_states[group]

            critic_out = self.critics[group](expand_joint_obs, expand_joint_act, **critic_kwargs)

            group_values = critic_out.values.reshape(*batch_shape, -1)

            for i, agent_key in enumerate(group_agents):
                q_eval[agent_key] = group_values[:, i]

        return q_eval

    def Qtarget(
            self,
            joint_observations: Tensor,
            joint_actions: Tensor,
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Dict[str, Tensor]:
        q_target = {}
        input_shape = joint_observations.shape
        bs = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            group_agents = self.groups[group]
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            if self.use_rnn:
                expand_joint_obs = joint_observations.unsqueeze(1).expand(-1, n_agent, -1, -1).reshape(bs, seq_len, -1)
                expand_joint_act = joint_actions.unsqueeze(1).expand(-1, n_agent, -1, -1).reshape(bs, seq_len, -1)
            else:
                expand_joint_obs = joint_observations.unsqueeze(1).expand(-1, n_agent, -1).reshape(bs, -1)
                expand_joint_act = joint_actions.unsqueeze(1).expand(-1, n_agent, -1).reshape(bs, -1)

            target_critic_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                target_critic_kwargs["rnn_states"] = rnn_states[group]

            target_critic_out = self.target_critics[group](expand_joint_obs, expand_joint_act, **target_critic_kwargs)

            group_values = target_critic_out.values.reshape(*batch_shape, -1)

            for i, agent_key in enumerate(group_agents):
                q_target[agent_key] = group_values[:, i]

        return q_target


class IndependentSoftActorCritic(OffPolicyMultiAgentActorCritic):
    def forward(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> MultiAgentModelOutput:
        rnn_states_new, pi_dists, actions, log_probs = {}, {}, {}, {}
        input_shape = observations[self.group_keys[0]].shape
        bs = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            group_agents = self.groups[group]
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            actor_kwargs = {"agent_indices": agent_indices[group]}
            if self.use_rnn:
                actor_kwargs["rnn_states"] = rnn_states[group]

            actor_out = self.actors[group](observations[group], **actor_kwargs)

            policy_dist = actor_out.distributions
            group_actions, group_log_action_prob = policy_dist.activated_rsample_and_logprob()

            group_actions = group_actions.reshape(*batch_shape, -1)
            group_log_action_prob = group_log_action_prob.reshape(*batch_shape, -1)
            pi_dists[group] = policy_dist
            rnn_states_new[group] = actor_out.representations.rnn_states

            for i, agent_key in enumerate(group_agents):
                actions[agent_key] = group_actions[:, i]
                log_probs[agent_key] = group_log_action_prob[:, i]

        return MultiAgentModelOutput(
            actions=actions,
            log_probs=log_probs,
            distributions=pi_dists,
            actor_rnn_states=rnn_states_new
        )

    def Qpolicy(
            self,
            observations: Dict[str, Tensor] | Tensor,
            actions: Dict[str, Tensor] | Tensor,
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Tuple[Dict[str, Tensor], ...]:
        q_eval_1, q_eval_2 = {}, {}

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:

            critic_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                critic_kwargs["rnn_states_1"] = rnn_states_1[group]
                critic_kwargs["rnn_states_2"] = rnn_states_2[group]

            critic_out = self.critics[group](observations[group], actions[group], **critic_kwargs)

            q_eval_1[group] = critic_out.values_1
            q_eval_2[group] = critic_out.values_2

        return q_eval_1, q_eval_2

    def Qtarget(
            self,
            observations: Dict[str, Tensor] | Tensor,
            actions: Dict[str, Tensor] | Tensor,
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Dict[str, Tensor]:
        q_target = {}

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:

            critic_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                critic_kwargs["rnn_states_1"] = rnn_states_1[group]
                critic_kwargs["rnn_states_2"] = rnn_states_2[group]

            target_critic_out = self.target_critics[group](observations[group], actions[group], **critic_kwargs)

            group_values_1 = target_critic_out.values_1
            group_values_2 = target_critic_out.values_2
            q_target[group] = torch.min(group_values_1, group_values_2)

        return q_target

    def init_critic_rnn_states(self, batch_size: int) -> Tuple[Dict[str, RNN_State], ...] | None:
        rnn_states_1, rnn_states_2 = None, None
        if self.use_rnn:
            rnn_states_1, rnn_states_2 = {}, {}
            for group in self.group_keys:
                bs = batch_size * self.n_group_agents[group]
                rnn_states_1[group] = self.critics[group].representation_1.obs_representation.init_rnn_states(bs)
                rnn_states_2[group] = self.critics[group].representation_2.obs_representation.init_rnn_states(bs)
        return rnn_states_1, rnn_states_2

    def init_critic_rnn_states_item(self, i_env: int,
                                    rnn_states_1: Dict[str, RNN_State],
                                    rnn_states_2: Dict[str, RNN_State] = None) -> Tuple[Dict[str, RNN_State], ...]:
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        batch_index = [i_env, ]
        for group in self.group_keys:
            rnn_states_1[group] = self.critics[group].representation_1.obs_representation.init_rnn_states_item(
                batch_index, rnn_states_1[group])
            rnn_states_2[group] = self.critics[group].representation_2.obs_representation.init_rnn_states_item(
                batch_index, rnn_states_2[group])
        return rnn_states_1, rnn_states_2


class MultiAgentSoftActorCritic(IndependentSoftActorCritic):
    def Qpolicy(
            self,
            joint_observations: Tensor,
            joint_actions: Tensor,
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Tuple[Dict[str, Tensor], ...]:
        q_eval_1, q_eval_2 = {}, {}
        input_shape = joint_observations.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            group_agents = self.groups[group]
            n_agent = self.n_group_agents[group]
            bs = batch_size * n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            if self.use_rnn:
                expand_joint_obs = joint_observations.unsqueeze(1).expand(-1, n_agent, -1, -1).reshape(bs, seq_len, -1)
                expand_joint_act = joint_actions.unsqueeze(1).expand(-1, n_agent, -1, -1).reshape(bs, seq_len, -1)
            else:
                expand_joint_obs = joint_observations.unsqueeze(1).expand(-1, n_agent, -1).reshape(bs, -1)
                expand_joint_act = joint_actions.unsqueeze(1).expand(-1, n_agent, -1).reshape(bs, -1)

            critic_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                critic_kwargs["rnn_states_1"] = rnn_states_1[group]
                critic_kwargs["rnn_states_2"] = rnn_states_2[group]

            critic_out = self.critics[group](expand_joint_obs, expand_joint_act, **critic_kwargs)

            group_values_1 = critic_out.values_1.reshape(*batch_shape, -1)
            group_values_2 = critic_out.values_2.reshape(*batch_shape, -1)

            for i, agent_key in enumerate(group_agents):
                q_eval_1[agent_key] = group_values_1[:, i]
                q_eval_2[agent_key] = group_values_2[:, i]

        return q_eval_1, q_eval_2

    def Qtarget(
            self,
            joint_observations: Tensor,
            joint_actions: Tensor,
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Dict[str, Tensor]:
        q_target = {}
        input_shape = joint_observations.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            group_agents = self.groups[group]
            n_agent = self.n_group_agents[group]
            bs = batch_size * n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            if self.use_rnn:
                expand_joint_obs = joint_observations.unsqueeze(1).expand(-1, n_agent, -1, -1).reshape(bs, seq_len, -1)
                expand_joint_act = joint_actions.unsqueeze(1).expand(-1, n_agent, -1, -1).reshape(bs, seq_len, -1)
            else:
                expand_joint_obs = joint_observations.unsqueeze(1).expand(-1, n_agent, -1).reshape(bs, -1)
                expand_joint_act = joint_actions.unsqueeze(1).expand(-1, n_agent, -1).reshape(bs, -1)

            target_critic_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                target_critic_kwargs["rnn_states_1"] = rnn_states_1[group]
                target_critic_kwargs["rnn_states_2"] = rnn_states_2[group]

            target_critic_out = self.target_critics[group](expand_joint_obs, expand_joint_act, **target_critic_kwargs)

            group_values_1 = target_critic_out.values_1.reshape(*batch_shape, -1)
            group_values_2 = target_critic_out.values_2.reshape(*batch_shape, -1)
            group_q_target = torch.min(group_values_1, group_values_2)

            for i, agent_key in enumerate(group_agents):
                q_target[agent_key] = group_q_target[:, i]

        return q_target


class IndependentTwinDelayedActorCritic(IndependentDeterministicActorCritic):
    def Qpolicy(
            self,
            observations: Dict[str, Tensor] | Tensor,
            actions: Dict[str, Tensor] | Tensor,
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Tuple[Dict[str, Tensor], ...]:
        q_eval_1, q_eval_2, q_eval = {}, {}, {}

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:

            critic_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                critic_kwargs["rnn_states_1"] = rnn_states_1[group]
                critic_kwargs["rnn_states_2"] = rnn_states_2[group]

            critic_out = self.critics[group](observations[group], actions[group], **critic_kwargs)

            q_eval_1[group] = critic_out.values_1
            q_eval_2[group] = critic_out.values_2
            q_eval[group] = (q_eval_1[group] + q_eval_2[group]) / 2.0

        return q_eval_1, q_eval_2, q_eval

    def Qtarget(
            self,
            observations: Dict[str, Tensor] | Tensor,
            actions: Dict[str, Tensor] | Tensor,
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Dict[str, Tensor]:
        q_target = {}

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:

            critic_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                critic_kwargs["rnn_states_1"] = rnn_states_1[group]
                critic_kwargs["rnn_states_2"] = rnn_states_2[group]

            target_critic_out = self.target_critics[group](observations[group], actions[group], **critic_kwargs)

            group_values_1 = target_critic_out.values_1
            group_values_2 = target_critic_out.values_2
            q_target[group] = torch.min(group_values_1, group_values_2)

        return q_target

    def init_critic_rnn_states(self, batch_size: int) -> Tuple[Dict[str, RNN_State], ...] | None:
        rnn_states_1, rnn_states_2 = None, None
        if self.use_rnn:
            rnn_states_1, rnn_states_2 = {}, {}
            for group in self.group_keys:
                bs = batch_size * self.n_group_agents[group]
                rnn_states_1[group] = self.critics[group].representation_1.obs_representation.init_rnn_states(bs)
                rnn_states_2[group] = self.critics[group].representation_2.obs_representation.init_rnn_states(bs)
        return rnn_states_1, rnn_states_2

    def init_critic_rnn_states_item(self, i_env: int,
                                    rnn_states_1: Dict[str, RNN_State],
                                    rnn_states_2: Dict[str, RNN_State] = None) -> Tuple[Dict[str, RNN_State], ...]:
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        batch_index = [i_env, ]
        for group in self.group_keys:
            rnn_states_1[group] = self.critics[group].representation_1.obs_representation.init_rnn_states_item(
                batch_index, rnn_states_1[group])
            rnn_states_2[group] = self.critics[group].representation_2.obs_representation.init_rnn_states_item(
                batch_index, rnn_states_2[group])
        return rnn_states_1, rnn_states_2


class MultiAgentTwinDelayedActorCritic(IndependentTwinDelayedActorCritic):
    def Qpolicy(
            self,
            joint_observations: Tensor,
            joint_actions: Tensor,
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Tuple[Dict[str, Tensor], ...]:
        q_eval_1, q_eval_2, q_eval = {}, {}, {}
        input_shape = joint_observations.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            bs = batch_size * n_agent

            if self.use_rnn:
                expand_joint_obs = joint_observations.unsqueeze(1).expand(-1, n_agent, -1, -1).reshape(bs, seq_len, -1)
                expand_joint_act = joint_actions.unsqueeze(1).expand(-1, n_agent, -1, -1).reshape(bs, seq_len, -1)
            else:
                expand_joint_obs = joint_observations.unsqueeze(1).expand(-1, n_agent, -1).reshape(bs, -1)
                expand_joint_act = joint_actions.unsqueeze(1).expand(-1, n_agent, -1).reshape(bs, -1)

            critic_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                critic_kwargs["rnn_states_1"] = rnn_states_1[group]
                critic_kwargs["rnn_states_2"] = rnn_states_2[group]

            critic_out = self.critics[group](expand_joint_obs, expand_joint_act, **critic_kwargs)

            q_eval_1[group] = critic_out.values_1
            q_eval_2[group] = critic_out.values_2
            q_eval[group] = (q_eval_1[group] + q_eval_2[group]) / 2.0

        return q_eval_1, q_eval_2, q_eval

    def Qtarget(
            self,
            joint_observations: Tensor,
            joint_actions: Tensor,
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Dict[str, Tensor]:
        q_target = {}
        input_shape = joint_observations.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            bs = batch_size * n_agent

            if self.use_rnn:
                expand_joint_obs = joint_observations.unsqueeze(1).expand(-1, n_agent, -1, -1).reshape(bs, seq_len, -1)
                expand_joint_act = joint_actions.unsqueeze(1).expand(-1, n_agent, -1, -1).reshape(bs, seq_len, -1)
            else:
                expand_joint_obs = joint_observations.unsqueeze(1).expand(-1, n_agent, -1).reshape(bs, -1)
                expand_joint_act = joint_actions.unsqueeze(1).expand(-1, n_agent, -1).reshape(bs, -1)

            target_critic_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                target_critic_kwargs["rnn_states_1"] = rnn_states_1[group]
                target_critic_kwargs["rnn_states_2"] = rnn_states_2[group]

            target_critic_out = self.target_critics[group](expand_joint_obs, expand_joint_act, **target_critic_kwargs)

            group_values_1 = target_critic_out.values_1
            group_values_2 = target_critic_out.values_2
            q_target[group] = torch.min(group_values_1, group_values_2)

        return q_target
