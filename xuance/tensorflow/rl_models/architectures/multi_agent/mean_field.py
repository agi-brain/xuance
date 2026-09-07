from typing import Dict, Optional, Union

from xuance.common import AgentGrouping
from xuance.tensorflow import tf, Tensor, Module, ModuleDict
from xuance.tensorflow.utils import AgentGroupedTensor
from xuance.tensorflow.rl_models.modules import MultiAgentModelOutput, RNN_State
from xuance.tensorflow.rl_models.architectures.multi_agent.value_factorization import MixingQNetwork
from xuance.tensorflow.rl_models.architectures.multi_agent.actor_critic import IndependentActorCritic


class MeanFieldQNetwork(MixingQNetwork):
    def __init__(self,
                 grouping: AgentGrouping,
                 q_networks: ModuleDict,
                 mixer: Module,
                 use_rnn: bool = False,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MeanFieldQNetwork, self).__init__(
            grouping=grouping,
            q_networks=q_networks,
            mixer=mixer,
            use_rnn=use_rnn,
            use_distributed_training=use_distributed_training,
            **kwargs
        )
        # The choice of policy: Boltzmann policy or greedy policy. (Default is 'greedy')
        self.n_actions_max = kwargs['n_actions_max']
        self.policy_type = kwargs['policy_type']
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self.temperature = kwargs['temperature']

    def call(
            self,
            observations: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            mean_actions: AgentGroupedTensor | None = None,
            avail_actions: AgentGroupedTensor | None = None,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None
    ) -> MultiAgentModelOutput:
        rep_out, rnn_states_new, actions, evalQ = {}, {}, {}, {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            individual_output = self.individual_q_networks[group](
                observations.packed(group),
                mean_actions.packed(group),
                agent_indices=agent_indices.packed(group),
                rnn_states=rnn_states[group] if self.use_rnn else None
            )

            rnn_states_new[group] = individual_output.representations.rnn_states
            rep_out[group] = individual_output.representations
            # Q value shape: batch_size * n_agent * -1 or batch_size * n_agent * seq_len * -1
            evalQ[group] = tf.reshape(individual_output.values, (*batch_shape, -1))

            evalQ_detach = tf.stop_gradient(evalQ[group])
            if avail_actions is not None:
                evalQ_detach = tf.where(avail_actions.group(group) == 0,
                                        tf.cast(-1e10, evalQ_detach.dtype),
                                        evalQ_detach)

            if self.policy_type == "Boltzmann":
                actions_prob = self.get_boltzmann_policy(evalQ_detach)
                shape = tf.shape(actions_prob)
                actions_prob_flat = tf.reshape(actions_prob, [-1, shape[-1]])
                actions_flat = tf.random.categorical(tf.math.log(actions_prob_flat), num_samples=1)
                actions[group] = tf.reshape(actions_flat, shape[:-1])
            elif self.policy_type == "greedy":
                actions[group] = tf.argmax(evalQ_detach, axis=-1, output_type=tf.int32)
            else:
                raise NotImplementedError

        return MultiAgentModelOutput(
            actions=AgentGroupedTensor(actions, self.grouping),
            values=AgentGroupedTensor(evalQ, self.grouping),
            rnn_states=rnn_states_new,
            rep_out=rep_out
        )

    def Qtarget(
            self,
            observations: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            mean_actions: AgentGroupedTensor | None = None,
            avail_actions: AgentGroupedTensor | None = None,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None
    ) -> MultiAgentModelOutput:
        rep_out, rnn_states_new, actions, targetQ = {}, {}, {}, {}
        input_shape = tf.shape(observations.grouped_tensor[self.group_keys[0]])
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            individual_output = self.target_individual_q_networks[group](
                observations.packed(group),
                mean_actions.packed(group),
                agent_indices=agent_indices.packed(group),
                rnn_states=rnn_states[group] if self.use_rnn else None
            )

            rnn_states_new[group] = individual_output.representations.rnn_states
            rep_out[group] = individual_output.representations
            targetQ[group] = tf.reshape(individual_output.values, (*batch_shape, -1))

            targetQ_detach = tf.stop_gradient(targetQ[group])
            if avail_actions is not None:
                targetQ_detach = tf.where(avail_actions.group(group) == 0,
                                          tf.cast(-1e10, targetQ_detach.dtype),
                                          targetQ_detach)

            if self.policy_type == "Boltzmann":
                actions_prob = self.get_boltzmann_policy(targetQ_detach)
                shape = tf.shape(actions_prob)
                actions_prob_flat = tf.reshape(actions_prob, [-1, shape[-1]])
                actions_flat = tf.random.categorical(tf.math.log(actions_prob_flat), num_samples=1)
                actions[group] = tf.reshape(actions_flat, shape[:-1])
            elif self.policy_type == "greedy":
                actions[group] = tf.argmax(targetQ_detach, axis=-1, output_type=tf.int32)
            else:
                raise NotImplementedError

        return MultiAgentModelOutput(
            actions=AgentGroupedTensor(actions, self.grouping),
            values=AgentGroupedTensor(targetQ, self.grouping),
            rnn_states=rnn_states_new,
            rep_out=rep_out
        )

    def get_boltzmann_policy(self, q):
        actions_prob = self.softmax(q / self.temperature)
        return actions_prob

    def get_mean_actions(self,
                         actions: Dict[str, Tensor],  # Dict of agent-wise tensor
                         agent_mask_tensor: Tensor, batch_size: int) -> Dict[str, Tensor]:
        masked_mean_actions_dict = {}
        actions_tensor = tf.reshape(tf.stack([v for v in actions.values()], axis=-1), [-1, self.n_agents])
        actions_onehot = tf.one_hot(actions_tensor, depth=self.n_actions_max)

        # count alive neighbors
        _eyes = tf.repeat(tf.expand_dims(tf.eye(self.n_agents, dtype=tf.float32), axis=0),
                          repeats=batch_size, axis=0)
        agent_mask_diagonal = tf.repeat(tf.expand_dims(agent_mask_tensor, axis=-1),
                                        repeats=self.n_agents, axis=2) * _eyes
        agent_mask_neighbors = tf.repeat(tf.expand_dims(agent_mask_tensor, axis=-1),
                                         repeats=self.n_agents, axis=2) - agent_mask_diagonal
        agent_alive_neighbors = tf.reduce_sum(agent_mask_neighbors, axis=-1, keepdims=True)

        # calculate mean actions of each agent's neighbors
        agent_mask_repeat = tf.repeat(tf.expand_dims(agent_mask_tensor, axis=-1), repeats=self.n_actions_max, axis=2)
        actions_onehot = actions_onehot * agent_mask_repeat
        actions_sum = tf.repeat(tf.reduce_sum(actions_onehot, axis=-2, keepdims=True), repeats=self.n_agents, axis=1)
        actions_neighbors_sum = actions_sum - actions_onehot  # Sum of other agents' actions.
        actions_mean_masked = actions_neighbors_sum * agent_mask_repeat / agent_alive_neighbors
        for i, agent_key in enumerate(self.agent_keys):
            masked_mean_actions_dict[agent_key] = actions_mean_masked[:, i]
        return masked_mean_actions_dict

    def copy_target(self):
        for ep, tp in zip(self.individual_q_networks.variables, self.target_individual_q_networks.variables):
            tp.assign(ep)


class MeanFiledActorCritic(IndependentActorCritic):
    def __init__(self,
                 grouping: AgentGrouping,
                 actors: ModuleDict,
                 critics: Module | ModuleDict,
                 use_rnn: bool = False,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MeanFiledActorCritic, self).__init__(
            grouping=grouping,
            actors=actors,
            critics=critics,
            use_rnn=use_rnn,
            use_distributed_training=use_distributed_training,
            **kwargs
        )
        self.n_actions_max = kwargs['n_actions_max']
        self.softmax = tf.nn.Softmax(dim=-1)
        self.temperature = kwargs['temperature']

    def get_boltzmann_policy(self, logits: Tensor) -> Tensor:
        """Convert Q-values to a Boltzmann (softmax) policy distribution.

        Args:
            logits (Tensor): Q-value tensor of shape [..., n_actions].

        Returns:
            Tensor: Probability distribution over actions with same shape as `q`.
        """
        actions_prob = self.softmax(logits / self.temperature)
        return actions_prob

    def call(
            self,
            observations: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            avail_actions: AgentGroupedTensor | None = None,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            deterministic: bool = False
    ) -> MultiAgentModelOutput:
        rnn_states_new, pi_dists, actions = {}, {}, {}
        input_shape = observations.grouped_tensor[self.group_keys[0]].shape
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            actor_kwargs = {
                "agent_indices": agent_indices.packed(group)
            }
            if avail_actions is not None:
                actor_kwargs["avail_actions"] = avail_actions.packed(group)
            if self.use_rnn:
                actor_kwargs["rnn_states"] = rnn_states[group]

            actor_out = self.actors[group](observations.packed(group), **actor_kwargs)

            pi_logits = actor_out.distributions.logits
            pi_probs = self.get_boltzmann_policy(pi_logits)
            self.actors[group].actor_head.policy_distribution.set_param(probs=pi_probs)
            policy_dist = self.actors[group].actor_head.policy_distribution

            if deterministic:
                sampled_actions = policy_dist.deterministic_sample()
            else:
                sampled_actions = policy_dist.stochastic_sample()

            actions[group] = sampled_actions.reshape(*batch_shape)

            rnn_states_new[group] = actor_out.representations.rnn_states
            pi_dists[group] = policy_dist

        return MultiAgentModelOutput(
            actions=AgentGroupedTensor(actions, self.grouping),
            distributions=pi_dists,
            actor_rnn_states=rnn_states_new
        )

    def get_mean_actions(self,
                         actions: Dict[str, Tensor],
                         agent_mask_tensor: Tensor, batch_size: int) -> Dict[str, Tensor]:
        masked_mean_actions_dict = {}
        actions_tensor = torch.stack([v for v in actions.values()], dim=-1).reshape([-1, self.n_agents])
        actions_onehot = F.one_hot(actions_tensor, num_classes=self.n_actions_max)

        # count alive neighbors
        _eyes = torch.eye(self.n_agents).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        agent_mask_diagonal = agent_mask_tensor.unsqueeze(-1).repeat(1, 1, self.n_agents) * _eyes
        agent_mask_neighbors = agent_mask_tensor.unsqueeze(-1).repeat(1, 1, self.n_agents) - agent_mask_diagonal
        agent_alive_neighbors = agent_mask_neighbors.sum(dim=-1, keepdim=True)

        # calculate mean actions of each agent's neighbors
        agent_mask_repeat = agent_mask_tensor.unsqueeze(-1).repeat(1, 1, self.n_actions_max)
        actions_onehot = actions_onehot * agent_mask_repeat
        actions_sum = actions_onehot.sum(dim=-2, keepdim=True).repeat(1, self.n_agents, 1)
        actions_neighbors_sum = actions_sum - actions_onehot  # Sum of other agents' actions.
        actions_mean_masked = actions_neighbors_sum * agent_mask_repeat / agent_alive_neighbors
        for i, agent_key in enumerate(self.agent_keys):
            masked_mean_actions_dict[agent_key] = actions_mean_masked[:, i]
        return masked_mean_actions_dict

    def get_values(
            self,
            observations: AgentGroupedTensor,
            agent_indices: AgentGroupedTensor,
            mean_actions: AgentGroupedTensor | None = None,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> MultiAgentModelOutput:
        rnn_states_new, values = {}, {}
        input_shape = observations.grouped_tensor[self.group_keys[0]].shape
        batch_size = input_shape[0]
        seq_len = input_shape[2] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agent = self.n_group_agents[group]
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            critic_kwargs = {
                "agent_indices": agent_indices.packed(group)
            }
            if self.use_rnn:
                critic_kwargs["rnn_states"] = rnn_states[group]

            critic_out = self.critics[group](observations.packed(group),
                                             mean_actions.packed(group),
                                             **critic_kwargs)

            values[group] = critic_out.values.reshape(*batch_shape, 1)
            rnn_states_new[group] = critic_out.representations.rnn_states

        return MultiAgentModelOutput(
            values=AgentGroupedTensor(values, self.grouping),
            critic_rnn_states=rnn_states_new
        )
