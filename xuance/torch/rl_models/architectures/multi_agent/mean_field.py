import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union
from torch import Tensor
from torch.nn import Module, ModuleDict
from torch.distributions import Categorical
from xuance.common import AgentGrouping
from xuance.torch.rl_models.modules import MultiAgentModelOutput, RNN_State
from xuance.torch.rl_models.architectures.multi_agent.value_factorization import MixingQNetwork
from xuance.torch.rl_models.architectures.multi_agent.actor_critic import IndependentActorCritic


class MeanFieldQNetwork(MixingQNetwork):
    def __init__(self,
                 grouping: AgentGrouping,
                 q_networks: ModuleDict,
                 mixer: Module,
                 use_rnn: bool = False,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MeanFieldQNetwork, self).__init__(
            grouping=grouping,
            q_networks=q_networks,
            mixer=mixer,
            use_rnn=use_rnn,
            device=device,
            use_distributed_training=use_distributed_training,
            **kwargs
        )
        # The choice of policy: Boltzmann policy or greedy policy. (Default is 'greedy')
        self.n_actions_max = kwargs['n_actions_max']
        self.policy_type = kwargs['policy_type']
        self.softmax = torch.nn.Softmax(dim=-1)
        self.temperature = kwargs['temperature']

    def forward(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            mean_actions: Dict[str, Tensor] = None,
            avail_actions: Dict[str, Tensor] = None,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None
    ) -> MultiAgentModelOutput:
        rep_out, rnn_states_new, actions, evalQ = {}, {}, {}, {}
        input_shape = observations[self.group_keys[0]].shape
        bs = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            group_agents = self.groups[group]
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            input_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                input_kwargs["rnn_states"] = rnn_states[group]

            individual_output = self.individual_q_networks[group](observations[group], mean_actions[group],
                                                                  **input_kwargs)

            rnn_states_new[group] = individual_output.representations.rnn_states
            rep_out[group] = individual_output.representations
            evalQ[group] = individual_output.values  # shape: bs * -1 or bs * seq_len * -1

            evalQ_detach = individual_output.values.reshape(*batch_shape, -1).clone().detach()
            if avail_actions is not None:
                evalQ_detach[avail_actions[group] == 0] = -1e10

            if self.policy_type == "Boltzmann":
                actions_prob = self.get_boltzmann_policy(evalQ_detach)
                group_actions = Categorical(probs=actions_prob).sample()
            elif self.policy_type == "greedy":
                group_actions = evalQ_detach.argmax(dim=-1, keepdim=False)
            else:
                raise NotImplementedError

            for i, agent_key in enumerate(group_agents):
                actions[agent_key] = group_actions[:, i]

        return MultiAgentModelOutput(
            actions=actions,
            values=evalQ,
            rnn_states=rnn_states_new,
            rep_out=rep_out
        )

    def Qtarget(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            mean_actions: Dict[str, Tensor] = None,
            avail_actions: Dict[str, Tensor] = None,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None
    ) -> MultiAgentModelOutput:
        rep_out, rnn_states_new, actions, targetQ = {}, {}, {}, {}
        input_shape = observations[self.group_keys[0]].shape
        bs = input_shape[0]
        seq_len = input_shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            group_agents = self.groups[group]
            n_agent = self.n_group_agents[group]
            batch_size = bs // n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            target_input_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                target_input_kwargs["rnn_states"] = rnn_states[group]

            individual_output = self.target_individual_q_networks[group](observations[group], mean_actions[group],
                                                                         **target_input_kwargs)

            rnn_states_new[group] = individual_output.representations.rnn_states
            rep_out[group] = individual_output.representations
            targetQ[group] = individual_output.values

            targetQ_detach = individual_output.values.reshape(*batch_shape, -1).clone().detach()
            if avail_actions is not None:
                targetQ_detach[avail_actions[group] == 0] = -1e10

            if self.policy_type == "Boltzmann":
                actions_prob = self.get_boltzmann_policy(targetQ_detach)
                group_actions = Categorical(probs=actions_prob).sample()
            elif self.policy_type == "greedy":
                group_actions = targetQ_detach.argmax(dim=-1, keepdim=False)
            else:
                raise NotImplementedError

            for i, agent_key in enumerate(group_agents):
                actions[agent_key] = group_actions[:, i]

        return MultiAgentModelOutput(
            actions=actions,
            values=targetQ,
            rnn_states=rnn_states_new,
            rep_out=rep_out
        )

    def get_boltzmann_policy(self, q):
        actions_prob = self.softmax(q / self.temperature)
        return actions_prob

    def get_mean_actions(self, actions: Dict[str, Tensor],
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

    def copy_target(self):
        for ep, tp in zip(self.individual_q_networks.parameters(), self.target_individual_q_networks.parameters()):
            tp.data.copy_(ep)


class MeanFiledActorCritic(IndependentActorCritic):
    def __init__(self,
                 grouping: AgentGrouping,
                 actors: ModuleDict,
                 critics: Module | ModuleDict,
                 use_rnn: bool = False,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MeanFiledActorCritic, self).__init__(
            grouping=grouping,
            actors=actors,
            critics=critics,
            use_rnn=use_rnn,
            device=device,
            use_distributed_training=use_distributed_training,
            **kwargs
        )
        self.n_actions_max = kwargs['n_actions_max']
        self.softmax = torch.nn.Softmax(dim=-1)
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

            pi_logits = actor_out.distributions.logits
            pi_probs = self.get_boltzmann_policy(pi_logits)
            self.actors[group].actor_head.policy_distribution.set_param(probs=pi_probs)
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

    def get_mean_actions(self, actions: Dict[str, Tensor],
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
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            mean_actions: Dict[str, Tensor] = None,
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

            critic_out = self.critics[group](observations[group], mean_actions[group], **critic_kwargs)

            group_values = critic_out.values.reshape(*batch_shape, 1)
            rnn_states_new[group] = critic_out.representations.rnn_states

            for i, agent_key in enumerate(group_agents):
                values[agent_key] = group_values[:, i]

        return MultiAgentModelOutput(values=values, critic_rnn_states=rnn_states_new)
