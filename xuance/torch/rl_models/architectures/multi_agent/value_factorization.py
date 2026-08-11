import os
import torch
import torch.nn.functional as F
from operator import itemgetter
from copy import deepcopy
from typing import Dict, Optional, Union, Tuple
from gymnasium.spaces import Discrete
from torch import Tensor
from torch.nn import Module, ModuleDict
from torch.nn.parallel import DistributedDataParallel
from xuance.common import AgentGrouping
from xuance.torch.rl_models.heads import QTRAN_Base, QTRAN_Alt, Coordination_Graph
from xuance.torch.rl_models.modules import MultiAgentModelOutput, RNN_State


class MixingQNetwork(Module):
    def __init__(self,
                 grouping: AgentGrouping,
                 q_networks: ModuleDict,
                 mixer: Module,
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

        self.individual_q_networks = q_networks
        self.target_individual_q_networks = deepcopy(self.individual_q_networks)
        self.eval_Qtot = mixer
        self.target_Qtot = deepcopy(self.eval_Qtot)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            self.individual_q_networks = DistributedDataParallel(module=self.individual_q_networks,
                                                                 device_ids=[self.rank])
            self.eval_Qtot = DistributedDataParallel(module=self.eval_Qtot, device_ids=[self.rank])

    @property
    def parameters_model(self):
        return list(self.individual_q_networks.parameters()) + list(self.eval_Qtot.parameters())

    def forward(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            avail_actions: Dict[str, Tensor] = None,
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
    ) -> MultiAgentModelOutput:
        rep_out, rnn_states_new, argmax_action, evalQ = {}, {}, {}, {}
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

            individual_output = self.individual_q_networks[group](observations[group], **input_kwargs)

            rnn_states_new[group] = individual_output.representations.rnn_states
            rep_out[group] = individual_output.representations
            evalQ[group] = individual_output.values  # shape: bs * -1 or bs * seq_len * -1

            # shape: batch_size * n_agent * -1 or batch_size * n_agent * seq_len * -1
            evalQ_detach = individual_output.values.reshape(*batch_shape, -1).clone().detach()
            if avail_actions is not None:
                evalQ_detach[avail_actions[group] == 0] = -1e10
            group_actions = evalQ_detach.argmax(dim=-1, keepdim=False)

            for i, agent_key in enumerate(group_agents):
                argmax_action[agent_key] = group_actions[:, i]  # get agent-wise actions for execution

        return MultiAgentModelOutput(
            actions=argmax_action,  # agent-wise
            values=evalQ,  # group-wise
            rnn_states=rnn_states_new,  # group-wise
            rep_out=rep_out  # group-wise
        )

    def Qtarget(self,
                observations: Dict[str, Tensor],
                agent_indices: Dict[str, Tensor],
                group_key: Optional[str] = None,
                rnn_states: Dict[str, RNN_State | dict] = None) -> MultiAgentModelOutput:
        rep_out, rnn_states_new, q_target = {}, {}, {}

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:

            target_input_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                target_input_kwargs["rnn_states"] = rnn_states[group]

            individual_output = self.target_individual_q_networks[group](observations[group],
                                                                         **target_input_kwargs)

            rnn_states_new[group] = individual_output.representations.rnn_states
            rep_out[group] = individual_output.representations
            q_target[group] = individual_output.values

        return MultiAgentModelOutput(
            values=q_target,  # group-wise
            rnn_states=rnn_states_new,  # group-wise
            rep_out=rep_out  # group-wise
        )

    def Q_tot(self, individual_values: Dict[str, Tensor], states: Optional[Tensor] = None):
        # Expected shape: [tot_batch_size * 1, ...] -> tot_batch_size * n_agents_all
        individual_inputs = torch.concat([individual_values[k].reshape([-1, 1]) for k in self.agent_keys], dim=-1)
        # Output shape: tot_batch_size * 1
        evalQ_tot = self.eval_Qtot(individual_inputs, states)
        return evalQ_tot

    def Qtarget_tot(self, individual_values: Dict[str, Tensor], states: Optional[Tensor] = None):
        # Expected shape: [tot_batch_size * 1, ...] -> tot_batch_size * n_agents_all
        individual_inputs = torch.concat([individual_values[k].reshape([-1, 1]) for k in self.agent_keys], dim=-1)
        # Output shape: tot_batch_size * 1
        q_target_tot = self.target_Qtot(individual_inputs, states)
        return q_target_tot

    def init_rnn_states(self, batch_size: int) -> Dict[str, RNN_State] | None:
        rnn_states = None
        if self.use_rnn:
            rnn_states = {}
            for group in self.group_keys:
                bs = batch_size * self.n_group_agents[group]
                rnn_states[group] = self.individual_q_networks[
                    group].representation.obs_representation.init_rnn_states(bs)
        return rnn_states

    def init_rnn_states_item(self, i_env: int,
                             rnn_states: Dict[str, RNN_State] = None) -> Dict[str, RNN_State]:
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        batch_index = [i_env, ]
        for group in self.group_keys:
            rnn_states[group] = self.individual_q_networks[
                group].representation.obs_representation.init_rnn_states_item(batch_index, rnn_states[group])
        return rnn_states

    def copy_target(self):
        for ep, tp in zip(self.individual_q_networks.parameters(), self.target_individual_q_networks.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qtot.parameters(), self.target_Qtot.parameters()):
            tp.data.copy_(ep)


class WeightedMixingQNetwork(MixingQNetwork):
    def __init__(self,
                 grouping: AgentGrouping,
                 q_networks: ModuleDict,
                 mixer: Module,
                 ff_mixer: Module,
                 use_rnn: bool = False,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(WeightedMixingQNetwork, self).__init__(grouping=grouping,
                                                     q_networks=q_networks,
                                                     mixer=mixer,
                                                     use_rnn=use_rnn,
                                                     device=device,
                                                     use_distributed_training=use_distributed_training,
                                                     **kwargs)
        self.individual_q_centralized = deepcopy(q_networks)
        self.target_individual_q_centralized = deepcopy(self.individual_q_centralized)
        self.ff_mixer = ff_mixer
        self.target_ff_mixer = deepcopy(self.ff_mixer)

        # Prepare DDP module.
        if self.distributed_training:
            self.individual_q_centralized = DistributedDataParallel(module=self.individual_q_centralized,
                                                                    device_ids=[self.rank])
            self.ff_mixer = DistributedDataParallel(module=self.ff_mixer, device_ids=[self.rank])

    @property
    def parameters_model(self):
        return list(self.individual_q_networks.parameters()) + list(self.eval_Qtot.parameters()) + list(
            self.individual_q_centralized.parameters()) + list(self.ff_mixer.parameters())

    def q_centralized(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None
    ) -> MultiAgentModelOutput:
        rnn_states_new, evalQ = {}, {}
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

            individual_output = self.individual_q_centralized[group](observations[group], **input_kwargs)

            individual_values = individual_output.values.reshape(*batch_shape, -1)
            rnn_states_new[group] = individual_output.representations.rnn_states

            for i, agent_key in enumerate(group_agents):
                evalQ[agent_key] = individual_values[:, i]

        return MultiAgentModelOutput(values=evalQ, rnn_states=rnn_states_new)

    def target_q_centralized(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None
    ) -> MultiAgentModelOutput:
        rnn_states_new, q_target = {}, {}
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

            individual_output = self.target_individual_q_centralized[group](observations[group], **target_input_kwargs)

            individual_values = individual_output.values.reshape(*batch_shape, -1)
            rnn_states_new[group] = individual_output.representations.rnn_states

            for i, agent_key in enumerate(group_agents):
                q_target[agent_key] = individual_values[:, i]

        return MultiAgentModelOutput(values=q_target, rnn_states=rnn_states_new)

    def q_feedforward(self, individual_values: Dict[str, Tensor], states: Optional[Tensor] = None):
        # Expected shape: [tot_batch_size * 1, ...] -> tot_batch_size * n_agents_all
        individual_inputs = torch.concat([individual_values[k].reshape([-1, 1]) for k in self.agent_keys], dim=-1)
        # Output shape: tot_batch_size * 1
        evalQ_tot = self.ff_mixer(individual_inputs, states)
        return evalQ_tot

    def target_q_feedforward(self, individual_values: Dict[str, Tensor], states: Optional[Tensor] = None):
        # Expected shape: [tot_batch_size * 1, ...] -> tot_batch_size * n_agents_all
        individual_inputs = torch.concat([individual_values[k].reshape([-1, 1]) for k in self.agent_keys], dim=-1)
        # Output shape: tot_batch_size * 1
        evalQ_tot = self.target_ff_mixer(individual_inputs, states)
        return evalQ_tot

    def init_centralized_rnn_states(self, batch_size: int) -> Dict[str, RNN_State] | None:
        rnn_states = None
        if self.use_rnn:
            rnn_states = {}
            for group in self.groups:
                bs = batch_size * self.n_group_agents[group]
                rnn_states[group] = self.individual_q_centralized[
                    group].representation.obs_representation.init_rnn_states(bs)
        return rnn_states

    def init_centralized_rnn_states_item(self, i_env: int,
                                         rnn_states: Dict[str, RNN_State] = None) -> Dict[str, RNN_State]:
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        batch_index = [i_env, ]
        for group in self.group_keys:
            rnn_states[group] = self.individual_q_centralized[
                group].representation.obs_representation.init_rnn_states_item(batch_index, rnn_states[group])
        return rnn_states

    def copy_target(self):
        super().copy_target()
        for ep, tp in zip(self.individual_q_centralized.parameters(),
                          self.target_individual_q_centralized.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.ff_mixer.parameters(), self.target_ff_mixer.parameters()):
            tp.data.copy_(ep)


class QTranMixingNetwork(MixingQNetwork):
    def __init__(self,
                 grouping: AgentGrouping,
                 q_networks: ModuleDict,
                 mixer: Module,
                 qtran_mixer: Union[QTRAN_Base, QTRAN_Alt],
                 use_rnn: bool = False,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(QTranMixingNetwork, self).__init__(grouping=grouping,
                                                 q_networks=q_networks,
                                                 mixer=mixer,
                                                 use_rnn=use_rnn,
                                                 device=device,
                                                 use_distributed_training=use_distributed_training,
                                                 **kwargs)
        self.n_actions = {k: self.individual_q_networks[k].action_space.n for k in self.group_keys}
        self.n_actions_max = max(self.n_actions.values())

        self.qtran_net = qtran_mixer
        self.target_qtran_net = deepcopy(qtran_mixer)

        # Prepare DDP module.
        if self.distributed_training:
            self.qtran_net = DistributedDataParallel(module=self.qtran_net, device_ids=[self.rank])

    @property
    def parameters_model(self):
        parameters_model = list(self.qtran_net.parameters()) + list(self.eval_Qtot.parameters()) + list(
            self.individual_q_networks.parameters())
        return parameters_model

    def Q_tran(self,
               states: Tensor,
               hidden_states: Dict[str, Tensor],
               actions: Dict[str, Tensor],
               agent_mask: Dict[str, Tensor] = None,
               avail_actions: Dict[str, Tensor] = None,
               group_key: Optional[str] = None) -> Tuple[Tensor, ...]:
        seq_len = states.shape[1] if self.use_rnn else 1
        batch_size = states.shape[0]
        hidden_states_dict, actions_onehot_dict = {}, {}
        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agents = self.n_group_agents[group]
            n_actions = self.n_actions[group]
            dim_hidden_state = hidden_states[group].shape[-1]
            group_actions_onehot = F.one_hot(actions[group].long(), n_actions)
            if self.use_rnn:
                actions_onehot_dict[group] = group_actions_onehot.reshape(batch_size, n_agents, seq_len, -1)
                hidden_states_dict[group] = hidden_states[group].reshape([-1, n_agents, seq_len, dim_hidden_state])
            else:
                actions_onehot_dict[group] = group_actions_onehot.reshape(batch_size, n_agents, -1)
                hidden_states_dict[group] = hidden_states[group].reshape([-1, n_agents, dim_hidden_state])

            if avail_actions is not None:
                actions_onehot_dict[group] *= avail_actions[group]
            if agent_mask is not None:
                if self.use_rnn:
                    agt_mask = agent_mask[group].reshape(
                        batch_size, n_agents, seq_len, 1).repeat(1, 1, 1, dim_hidden_state)
                else:
                    agt_mask = agent_mask[group].reshape(batch_size, n_agents, 1).repeat(1, 1, dim_hidden_state)
                hidden_states_dict[group] = hidden_states_dict[group] * agt_mask

        hidden_states_tensor_in = torch.concat([hidden_states_dict[k] for k in self.group_keys], dim=1)
        actions_onehot = torch.concat([actions_onehot_dict[k] for k in self.group_keys], dim=1)

        if self.use_rnn:
            states = states.reshape(batch_size * seq_len, -1)
            hidden_states_tensor_in = hidden_states_tensor_in.transpose(1, 2).reshape(-1, self.n_agents,
                                                                                      dim_hidden_state)
            actions_onehot = actions_onehot.transpose(1, 2).reshape(-1, self.n_agents, self.n_actions_max)
        q_jt, v_jt = self.qtran_net(states, hidden_states_tensor_in, actions_onehot)
        return q_jt, v_jt

    def Q_tran_target(self,
                      states: Tensor,
                      hidden_states: Dict[str, Tensor],
                      actions: Dict[str, Tensor],
                      agent_mask: Dict[str, Tensor] = None,
                      avail_actions: Dict[str, Tensor] = None,
                      group_key: Optional[str] = None) -> Tuple[Tensor, ...]:
        seq_len = states.shape[1] if self.use_rnn else 1
        batch_size = states.shape[0]
        hidden_states_dict, actions_onehot_dict = {}, {}
        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            n_agents = self.n_group_agents[group]
            n_actions = self.n_actions[group]
            dim_hidden_state = hidden_states[group].shape[-1]
            group_actions_onehot = F.one_hot(actions[group].long(), n_actions)
            if self.use_rnn:
                actions_onehot_dict[group] = group_actions_onehot.reshape(batch_size, n_agents, seq_len, -1)
                hidden_states_dict[group] = hidden_states[group].reshape([-1, n_agents, seq_len, dim_hidden_state])
            else:
                actions_onehot_dict[group] = group_actions_onehot.reshape(batch_size, n_agents, -1)
                hidden_states_dict[group] = hidden_states[group].reshape([-1, n_agents, dim_hidden_state])

            if avail_actions is not None:
                actions_onehot_dict[group] *= avail_actions[group]
            if agent_mask is not None:
                if self.use_rnn:
                    agt_mask = agent_mask[group].reshape(
                        batch_size, n_agents, seq_len, 1).repeat(1, 1, 1, dim_hidden_state)
                else:
                    agt_mask = agent_mask[group].reshape(batch_size, n_agents, 1).repeat(1, 1, dim_hidden_state)
                hidden_states_dict[group] = hidden_states_dict[group] * agt_mask

        hidden_states_tensor_in = torch.concat([hidden_states_dict[k] for k in self.group_keys], dim=1)
        actions_onehot = torch.concat([actions_onehot_dict[k] for k in self.group_keys], dim=1)

        if self.use_rnn:
            states = states.reshape(batch_size * seq_len, -1)
            hidden_states_tensor_in = hidden_states_tensor_in.transpose(1, 2).reshape(-1, self.n_agents,
                                                                                      dim_hidden_state)
            actions_onehot = actions_onehot.transpose(1, 2).reshape(-1, self.n_agents, self.n_actions_max)
        q_jt, v_jt = self.target_qtran_net(states, hidden_states_tensor_in, actions_onehot)
        return q_jt, v_jt

    def copy_target(self):
        super().copy_target()
        for ep, tp in zip(self.qtran_net.parameters(), self.target_qtran_net.parameters()):
            tp.data.copy_(ep)


class DeepCoordinationGraph(Module):
    def __init__(self,
                 grouping: AgentGrouping,
                 action_space: Discrete,
                 representation: ModuleDict,
                 utility: Module,
                 payoffs: Module,
                 dcg_graph: Coordination_Graph,
                 dcg_s: bool = False,
                 bias: Optional[Module] = None,
                 use_rnn: bool = False,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__()

        self.grouping = grouping
        self.groups = grouping.groups
        self.group_keys = grouping.group_keys
        self.agent_keys = grouping.agent_keys
        self.action_space = action_space
        self.n_agents = len(self.agent_keys)
        self.n_group_agents = {k: len(self.groups[k]) for k in self.group_keys}
        self.dcg_s = dcg_s
        self.use_rnn = use_rnn
        self.device = device

        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.utility = utility
        self.target_utility = deepcopy(utility)
        self.payoffs = payoffs
        self.target_payoffs = deepcopy(payoffs)
        self.bias = bias
        self.target_bias = deepcopy(bias)
        self.graph = dcg_graph

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            self.representation = DistributedDataParallel(module=self.representation, device_ids=[self.rank])
            self.utility = DistributedDataParallel(module=self.utility, device_ids=[self.rank])
            self.payoffs = DistributedDataParallel(module=self.payoffs, device_ids=[self.rank])
            if self.dcg_s:
                self.bias = DistributedDataParallel(module=self.bias, device_ids=[self.rank])

    @property
    def parameters_model(self):
        parameters_model = list(self.representation.parameters()) + \
                           list(self.utility.parameters()) + \
                           list(self.payoffs.parameters())
        if self.dcg_s:
            parameters_model += list(self.bias.parameters())
        return parameters_model

    def get_hidden_states(self,
                          observations: Dict[str, Tensor],
                          agent_indices: Dict[str, Tensor],
                          group_key: Optional[str] = None,
                          rnn_states: Dict[str, RNN_State | dict] = None,
                          use_target_net=False):
        """
        Get the hidden states of the representations for all agents.

        Args:
            observations: Dict[str, Tensor].
            agent_indices: Dict[str, Tensor]: Agent indices.
            rnn_states: Dict[str, RNN_State | dict]: The hidden variables of the RNN.
            use_target_net (bool): Whether to use a target network or not.

        Returns:
            rnn_hidden: The RNN hidden states for next step calculating.
            hidden_states_n: The hidden states of the representations that what we want.
        """
        rnn_states_new, hidden_states = {}, {}
        batch_size = observations[self.group_keys[0]].shape[0]
        seq_len = observations[self.group_keys[0]].shape[1] if self.use_rnn else 1

        group_list = self.group_keys if group_key is None else [group_key]

        for group in group_list:
            group_agents = self.groups[group]
            n_agent = self.n_group_agents[group]
            batch_size = batch_size // n_agent
            batch_shape = (batch_size, n_agent, seq_len) if self.use_rnn else (batch_size, n_agent)

            input_kwargs = {
                "agent_indices": agent_indices[group]
            }
            if self.use_rnn:
                input_kwargs["rnn_states"] = rnn_states[group]

            if use_target_net:
                representation_out = self.target_representation[group](observations[group], **input_kwargs)
            else:
                representation_out = self.representation[group](observations[group], **input_kwargs)

            hidden_states_out = representation_out.embeddings.reshape(*batch_shape, -1)
            rnn_states_new[group] = representation_out.rnn_states

            for i, agent_key in enumerate(group_agents):
                hidden_states[agent_key] = hidden_states_out[:, i]

        if self.use_rnn:
            hidden_states_n = torch.stack(itemgetter(*self.agent_keys)(hidden_states), dim=-2)
            hidden_states_n = hidden_states_n.reshape(batch_size, seq_len, self.n_agents, -1)
        else:
            hidden_states_n = torch.stack(itemgetter(*self.agent_keys)(hidden_states), dim=-2)
            hidden_states_n = hidden_states_n.reshape(batch_size, self.n_agents, -1)

        return rnn_states_new, hidden_states_n

    def init_rnn_states(self, batch_size: int) -> Dict[str, RNN_State] | None:
        rnn_states = None
        if self.use_rnn:
            rnn_states = {}
            for group in self.groups:
                bs = batch_size * self.n_group_agents[group]
                rnn_states[group] = self.representation[group].obs_representation.init_rnn_states(bs)
        return rnn_states

    def init_rnn_states_item(self, i_env: int,
                             rnn_states: Dict[str, RNN_State] = None) -> Dict[str, RNN_State]:
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        batch_index = [i_env, ]
        for group in self.group_keys:
            rnn_states[group] = self.representation[group].obs_representation.init_rnn_states_item(
                batch_index, rnn_states[group])
        return rnn_states

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.utility.parameters(), self.target_utility.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.payoffs.parameters(), self.target_payoffs.parameters()):
            tp.data.copy_(ep)
        if self.dcg_s:
            for ep, tp in zip(self.bias.parameters(), self.target_bias.parameters()):
                tp.data.copy_(ep)
