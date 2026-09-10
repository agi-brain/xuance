from copy import deepcopy
from typing import Dict, Optional, Tuple

from xuance.common import AgentGrouping
from xuance.tensorflow import tf, Tensor, Module, ModuleDict
from xuance.tensorflow.utils import AgentGroupedTensor
from xuance.tensorflow.rl_models.modules import RNN_State, MultiAgentModelOutput
from .base import OffPolicyMultiAgentActorCritic


class IndependentActorCritic(Module):
    def __init__(self,
                 grouping: AgentGrouping,
                 actors: ModuleDict,
                 critics: Module | ModuleDict,
                 use_rnn: bool = False,
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

        self.actors = actors
        self.critics = critics

        # Prepare DDP module.

    @property
    def parameters_model(self):
        return {
            group: list(self.actors[group].parameters()) + list(self.critics[group].parameters())
            for group in self.group_keys
        }

    def call(
            self,
            observations: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            avail_actions: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            deterministic: bool = False
    ) -> MultiAgentModelOutput:
        rnn_states_new, pi_dists, actions = {}, {}, {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            actor_out = self.actors[group](observations.packed(group),
                                           avail_actions=None if avail_actions is None else avail_actions.packed(group),
                                           agent_indices=agent_indices.packed(group),
                                           rnn_states=rnn_states[group] if self.use_rnn else None)

            policy_dist = actor_out.distributions
            if deterministic:
                sampled_actions = policy_dist.deterministic_sample()
            else:
                sampled_actions = policy_dist.stochastic_sample()
            actions[group] = tf.reshape(sampled_actions, (*batch_shape, -1))

            rnn_states_new[group] = actor_out.representations.rnn_states
            pi_dists[group] = actor_out.distributions

        return MultiAgentModelOutput(
            actions=AgentGroupedTensor(actions, self.grouping),
            distributions=pi_dists,
            actor_rnn_states=rnn_states_new
        )

    def get_values(
            self,
            observations: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            states: Tensor = None,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> MultiAgentModelOutput:
        rnn_states_new, values = {}, {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            critic_out = self.critics[group](observations.packed(group),
                                             agent_indices=agent_indices.packed(group),
                                             rnn_states=rnn_states[group] if self.use_rnn else None)

            values[group] = tf.reshape(critic_out.values, (*batch_shape, 1))

            rnn_states_new[group] = critic_out.representations.rnn_states

        return MultiAgentModelOutput(
            values=AgentGroupedTensor(values, self.grouping),
            critic_rnn_states=rnn_states_new
        )

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
            observations: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            states: Tensor = None,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> MultiAgentModelOutput:
        rnn_states_new, values = {}, {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            if states is not None:
                expanded_states = states.unsqueeze(1).repeat(1, self.n_agents, 1)  # batch * N * dim_S
            else:
                expanded_states = None

            critic_out = self.critics[group](observations=observations, agent_indices=agent_indices,
                                             expanded_states=expanded_states,
                                             rnn_states=rnn_states[group] if self.use_rnn else None)

            rnn_states_new[group] = critic_out.critic_rnn_states
            values[group] = tf.reshape(critic_out.values, (*batch_shape, 1))

        return MultiAgentModelOutput(
            values=AgentGroupedTensor(values, self.grouping),
            critic_rnn_states=rnn_states_new
        )

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

    def call(
            self,
            observations: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            avail_actions: AgentGroupedTensor | None = None,
            epsilon: float = 0.0,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            deterministic: bool = False,
            test_mode: bool = False
    ) -> MultiAgentModelOutput:
        rnn_states_new, pi_dists, actions = {}, {}, {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            actor_out = self.actors[group](observations.packed(group),
                                           agent_indices=agent_indices.packed(group),
                                           avail_actions=None if avail_actions is None else avail_actions.packed(group),
                                           rnn_states=rnn_states[group])

            group_probs = actor_out.distributions.probs

            if not test_mode:
                group_probs = (1 - epsilon) * group_probs + epsilon * 1 / self.actors[group].action_dim

            self.actors[group].actor_head.policy_distribution.set_param(probs=group_probs)
            policy_dist = self.actors[group].actor_head.policy_distribution

            if deterministic:
                sampled_actions = policy_dist.deterministic_sample()
            else:
                sampled_actions = policy_dist.stochastic_sample()

            actions[group] = tf.reshape(sampled_actions, (*batch_shape, -1))

            rnn_states_new[group] = actor_out.representations.rnn_states
            pi_dists[group] = policy_dist

        return MultiAgentModelOutput(
            actions=AgentGroupedTensor(actions, self.grouping),
            distributions=pi_dists,
            actor_rnn_states=rnn_states_new
        )

    def get_values(
            self,
            observations: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
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
            tp.assign(ep)


class ValueDecompositionActorCritic(IndependentActorCritic):
    def __init__(self,
                 grouping: AgentGrouping,
                 actors: ModuleDict,
                 critics: Module | ModuleDict,
                 mixer: Module,
                 use_rnn: bool = False,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__(
            grouping=grouping,
            actors=actors,
            critics=critics,
            use_rnn=use_rnn,
            use_distributed_training=use_distributed_training,
            **kwargs
        )
        self.v_tot = mixer

    def values_tot(self, individual_values: Dict[str, Tensor], global_states: Optional[Tensor] = None):
        # Expected shape: [tot_batch_size * 1, ...] -> tot_batch_size * n_agents_all
        individual_inputs = tf.concat([tf.reshape(individual_values[k], [-1, 1]) for k in self.agent_keys], axis=-1)
        # Output shape: tot_batch_size * 1
        values = self.v_tot(individual_inputs, global_states)
        return values


class IndependentDeterministicActorCritic(OffPolicyMultiAgentActorCritic):
    def call(
            self,
            observations: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> MultiAgentModelOutput:
        rnn_states_new, actions = {}, {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            actor_out = self.actors[group](observations.packed(group),
                                           agent_indices=agent_indices.packed(group),
                                           rnn_states=rnn_states[group] if self.use_rnn else None)

            rnn_states_new[group] = actor_out.representations.rnn_states
            actions[group] = tf.reshape(actor_out.actions, (*batch_shape, -1))

        return MultiAgentModelOutput(
            actions=AgentGroupedTensor(actions, self.grouping),
            actor_rnn_states=rnn_states_new
        )

    def Qpolicy(
            self,
            observations: AgentGroupedTensor,
            actions: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> AgentGroupedTensor:
        q_eval = {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            critic_out = self.critics[group](observations.packed(group),
                                             actions.packed(group),
                                             agent_indices=agent_indices.packed(group),
                                             rnn_states=rnn_states[group] if self.use_rnn else None)

            q_eval[group] = tf.reshape(critic_out.values, (*batch_shape, -1))

        return AgentGroupedTensor(q_eval, self.grouping)

    def Qtarget(
            self,
            observations: AgentGroupedTensor,
            actions: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> AgentGroupedTensor:
        q_target = {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            target_critic_out = self.target_critics[group](observations.packed(group),
                                                           actions.packed(group),
                                                           agent_indices=agent_indices.packed(group),
                                                           rnn_states=rnn_states[group] if self.use_rnn else None)

            q_target[group] = tf.reshape(target_critic_out.values, (*batch_shape, -1))

        return AgentGroupedTensor(q_target, self.grouping)

    def Atarget(
            self,
            observations: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> AgentGroupedTensor:
        actions = {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            target_actor_out = self.target_actors[group](observations.packed(group),
                                                         agent_indices=agent_indices.packed(group),
                                                         rnn_states=rnn_states[group] if self.use_rnn else None)

            actions[group] = tf.reshape(target_actor_out.actions, (*batch_shape, -1))

        return AgentGroupedTensor(actions, self.grouping)


class MultiAgentDeterministicActorCritic(IndependentDeterministicActorCritic):
    def Qpolicy(
            self,
            joint_observations: Tensor,
            joint_actions: Tensor,
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> AgentGroupedTensor:
        q_eval = {}
        input_shape = tf.shape(joint_observations)
        bs = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            expand_joint_obs = tf.repeat(tf.expand_dims(joint_observations, axis=1), repeats=n_agent, axis=1)
            expand_joint_obs = tf.reshape(expand_joint_obs, [bs, seq_len, -1] if self.use_rnn else [bs, -1])

            expand_joint_act = tf.repeat(tf.expand_dims(joint_actions, axis=1), repeats=n_agent, axis=1)
            expand_joint_act = tf.reshape(expand_joint_act, [bs, seq_len, -1] if self.use_rnn else [bs, -1])

            critic_out = self.critics[group](expand_joint_obs, expand_joint_act,
                                             agent_indices=agent_indices.packed(group),
                                             rnn_states=rnn_states[group] if self.use_rnn else None)

            q_eval[group] = tf.reshape(critic_out.values, (*batch_shape, -1))

        return AgentGroupedTensor(q_eval, self.grouping)

    def Qtarget(
            self,
            joint_observations: Tensor,
            joint_actions: Tensor,
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> AgentGroupedTensor:
        q_target = {}
        input_shape = tf.shape(joint_observations)
        bs = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            expand_joint_obs = tf.repeat(tf.expand_dims(joint_observations, axis=1), repeats=n_agent, axis=1)
            expand_joint_obs = tf.reshape(expand_joint_obs, [bs, seq_len, -1] if self.use_rnn else [bs, -1])

            expand_joint_act = tf.repeat(tf.expand_dims(joint_actions, axis=1), repeats=n_agent, axis=1)
            expand_joint_act = tf.reshape(expand_joint_act, [bs, seq_len, -1] if self.use_rnn else [bs, -1])

            target_critic_out = self.target_critics[group](expand_joint_obs, expand_joint_act,
                                                           agent_indices=agent_indices.packed(group),
                                                           rnn_states=rnn_states[group] if self.use_rnn else None)

            q_target[group] = tf.reshape(target_critic_out.values, (*batch_shape, -1))

        return AgentGroupedTensor(q_target, self.grouping)


class IndependentSoftActorCritic(OffPolicyMultiAgentActorCritic):
    def call(
            self,
            observations: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> MultiAgentModelOutput:
        rnn_states_new, pi_dists, actions, log_probs = {}, {}, {}, {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            actor_out = self.actors[group](observations.packed(group),
                                           agent_indices=agent_indices.packed(group),
                                           rnn_states=rnn_states[group] if self.use_rnn else None)

            policy_dist = actor_out.distributions
            group_actions, group_log_action_prob = policy_dist.activated_rsample_and_logprob()

            actions[group] = tf.reshape(group_actions, (*batch_shape, -1))
            log_probs[group] = tf.reshape(group_log_action_prob, (*batch_shape, -1))
            pi_dists[group] = policy_dist
            rnn_states_new[group] = actor_out.representations.rnn_states

        return MultiAgentModelOutput(
            actions=AgentGroupedTensor(actions, self.grouping),
            log_probs=AgentGroupedTensor(log_probs, self.grouping),
            distributions=pi_dists,
            actor_rnn_states=rnn_states_new
        )

    def Qpolicy(
            self,
            observations: AgentGroupedTensor,
            actions: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Tuple[AgentGroupedTensor, ...]:
        q_eval_1, q_eval_2 = {}, {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            critic_out = self.critics[group](observations.packed(group), actions.packed(group),
                                             agent_indices=agent_indices.packed(group),
                                             rnn_states_1=rnn_states_1[group] if self.use_rnn else None,
                                             rnn_states_2=rnn_states_2[group] if self.use_rnn else None)

            q_eval_1[group] = tf.reshape(critic_out.values_1, (*batch_shape, -1))
            q_eval_2[group] = tf.reshape(critic_out.values_2, (*batch_shape, -1))

        return AgentGroupedTensor(q_eval_1, self.grouping), AgentGroupedTensor(q_eval_2, self.grouping)

    def Qtarget(
            self,
            observations: AgentGroupedTensor,
            actions: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> AgentGroupedTensor:
        q_target = {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            target_critic_out = self.target_critics[group](observations.packed(group), actions.packed(group),
                                                           agent_indices=agent_indices.packed(group),
                                                           rnn_states_1=rnn_states_1[group] if self.use_rnn else None,
                                                           rnn_states_2=rnn_states_2[group] if self.use_rnn else None)

            group_values_1 = target_critic_out.values_1
            group_values_2 = target_critic_out.values_2
            q_target[group] = tf.reshape(tf.minimum(group_values_1, group_values_2), (*batch_shape, -1))

        return AgentGroupedTensor(q_target, self.grouping)

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

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critics.variables, self.target_critics.variables):
            tp.assign((1 - tau) * tp + tau * ep)


class MultiAgentSoftActorCritic(IndependentSoftActorCritic):
    def Qpolicy(
            self,
            joint_observations: Tensor,
            joint_actions: Tensor,
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Tuple[AgentGroupedTensor, ...]:
        q_eval_1, q_eval_2 = {}, {}
        input_shape = tf.shape(joint_observations)
        batch_size = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            bs = batch_size * n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            expand_joint_obs = tf.repeat(tf.expand_dims(joint_observations, axis=1), repeats=n_agent, axis=1)
            expand_joint_obs = tf.reshape(expand_joint_obs, [bs, seq_len, -1] if self.use_rnn else [bs, -1])

            expand_joint_act = tf.repeat(tf.expand_dims(joint_actions, axis=1), repeats=n_agent, axis=1)
            expand_joint_act = tf.reshape(expand_joint_act, [bs, seq_len, -1] if self.use_rnn else [bs, -1])

            critic_out = self.critics[group](expand_joint_obs, expand_joint_act,
                                             agent_indices=agent_indices.packed(group),
                                             rnn_states_1=rnn_states_1[group] if self.use_rnn else None,
                                             rnn_states_2=rnn_states_2[group] if self.use_rnn else None)

            q_eval_1[group] = tf.reshape(critic_out.values_1, (*batch_shape, -1))
            q_eval_2[group] = tf.reshape(critic_out.values_2, (*batch_shape, -1))

        return AgentGroupedTensor(q_eval_1, self.grouping), AgentGroupedTensor(q_eval_2, self.grouping)

    def Qtarget(
            self,
            joint_observations: Tensor,
            joint_actions: Tensor,
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> AgentGroupedTensor:
        q_target = {}
        input_shape = tf.shape(joint_observations)
        batch_size = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            bs = batch_size * n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            expand_joint_obs = tf.repeat(tf.expand_dims(joint_observations, axis=1), repeats=n_agent, axis=1)
            expand_joint_obs = tf.reshape(expand_joint_obs, [bs, seq_len, -1] if self.use_rnn else [bs, -1])

            expand_joint_act = tf.repeat(tf.expand_dims(joint_actions, axis=1), repeats=n_agent, axis=1)
            expand_joint_act = tf.reshape(expand_joint_act, [bs, seq_len, -1] if self.use_rnn else [bs, -1])

            target_critic_out = self.target_critics[group](expand_joint_obs, expand_joint_act,
                                                           agent_indices=agent_indices.packed(group),
                                                           rnn_states_1=rnn_states_1[group] if self.use_rnn else None,
                                                           rnn_states_2=rnn_states_2[group] if self.use_rnn else None)

            group_values_1 = target_critic_out.values_1
            group_values_2 = target_critic_out.values_2
            q_target[group] = tf.reshape(tf.minimum(group_values_1, group_values_2), (*batch_shape, -1))

        return AgentGroupedTensor(q_target, self.grouping)


class IndependentTwinDelayedActorCritic(IndependentDeterministicActorCritic):
    def Qpolicy(
            self,
            observations: AgentGroupedTensor,
            actions: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Tuple[AgentGroupedTensor, ...]:
        q_eval_1, q_eval_2, q_eval = {}, {}, {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            critic_out = self.critics[group](observations.packed(group), actions.packed(group),
                                             agent_indices=agent_indices,
                                             rnn_states_1=rnn_states_1[group] if self.use_rnn else None,
                                             rnn_states_2=rnn_states_2[group] if self.use_rnn else None)

            q_eval_1[group] = tf.reshape(critic_out.values_1, (*batch_shape, -1))
            q_eval_2[group] = tf.reshape(critic_out.values_2, (*batch_shape, -1))
            q_eval[group] = (q_eval_1[group] + q_eval_2[group]) / 2.0

        return (AgentGroupedTensor(q_eval_1, self.grouping),
                AgentGroupedTensor(q_eval_2, self.grouping),
                AgentGroupedTensor(q_eval, self.grouping))

    def Qtarget(
            self,
            observations: AgentGroupedTensor,
            actions: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> AgentGroupedTensor:
        q_target = {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            target_critic_out = self.target_critics[group](observations.packed(group), actions.packed(group),
                                                           agent_indices=agent_indices.packed(group),
                                                           rnn_states_1=rnn_states_1[group] if self.use_rnn else None,
                                                           rnn_states_2=rnn_states_2[group] if self.use_rnn else None)

            group_values_1 = target_critic_out.values_1
            group_values_2 = target_critic_out.values_2
            q_target[group] = tf.reshape(tf.minimum(group_values_1, group_values_2), (*batch_shape, -1))

        return AgentGroupedTensor(q_target, self.grouping)

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
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Tuple[AgentGroupedTensor, ...]:
        q_eval_1, q_eval_2, q_eval = {}, {}, {}
        input_shape = tf.shape(joint_observations)
        batch_size = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)
            bs = batch_size * n_agent

            expand_joint_obs = tf.repeat(tf.expand_dims(joint_observations, axis=1), repeats=n_agent, axis=1)
            expand_joint_obs = tf.reshape(expand_joint_obs, [bs, seq_len, -1] if self.use_rnn else [bs, -1])

            expand_joint_act = tf.repeat(tf.expand_dims(joint_actions, axis=1), repeats=n_agent, axis=1)
            expand_joint_act = tf.reshape(expand_joint_act, [bs, seq_len, -1] if self.use_rnn else [bs, -1])

            critic_out = self.critics[group](expand_joint_obs, expand_joint_act,
                                             agent_indices=agent_indices.packed(group),
                                             rnn_states_1=rnn_states_1[group] if self.use_rnn else None,
                                             rnn_states_2=rnn_states_2[group] if self.use_rnn else None)

            q_eval_1[group] = tf.reshape(critic_out.values_1, (*batch_shape, -1))
            q_eval_2[group] = tf.reshape(critic_out.values_2, (*batch_shape, -1))
            q_eval[group] = (q_eval_1[group] + q_eval_2[group]) / 2.0

        return (AgentGroupedTensor(q_eval_1, self.grouping),
                AgentGroupedTensor(q_eval_2, self.grouping),
                AgentGroupedTensor(q_eval, self.grouping))

    def Qtarget(
            self,
            joint_observations: Tensor,
            joint_actions: Tensor,
            agent_indices: AgentGroupedTensor,
            group_key: Optional[str] = None,
            rnn_states_1: Dict[str, RNN_State | dict] = None,
            rnn_states_2: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> AgentGroupedTensor:
        q_target = {}
        input_shape = tf.shape(joint_observations)
        batch_size = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)
            bs = batch_size * n_agent

            expand_joint_obs = tf.repeat(tf.expand_dims(joint_observations, axis=1), repeats=n_agent, axis=1)
            expand_joint_obs = tf.reshape(expand_joint_obs, [bs, seq_len, -1] if self.use_rnn else [bs, -1])

            expand_joint_act = tf.repeat(tf.expand_dims(joint_actions, axis=1), repeats=n_agent, axis=1)
            expand_joint_act = tf.reshape(expand_joint_act, [bs, seq_len, -1] if self.use_rnn else [bs, -1])

            target_critic_out = self.target_critics[group](expand_joint_obs, expand_joint_act,
                                                           agent_indices=agent_indices.packed(group),
                                                           rnn_states_1=rnn_states_1[group] if self.use_rnn else None,
                                                           rnn_states_2=rnn_states_2[group] if self.use_rnn else None)

            group_values_1 = target_critic_out.values_1
            group_values_2 = target_critic_out.values_2
            q_target[group] = tf.reshape(tf.minimum(group_values_1, group_values_2), (*batch_shape, -1))

        return AgentGroupedTensor(q_target, self.grouping)
