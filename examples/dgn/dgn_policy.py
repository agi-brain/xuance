from operator import itemgetter
from types import ModuleType
from typing import Optional, Dict, Sequence, Callable, Union, List

import torch
from gymnasium.spaces import Discrete
from torch import Tensor, nn
from torch.nn import ModuleDict

from xuance.torch.policies.deterministic_marl import BasicQnetwork


class DGN_Policy(BasicQnetwork):
    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation: ModuleDict,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 communicator: ModuleDict = None,
                 **kwargs):
        super(DGN_Policy, self).__init__(action_space=action_space,
                                         n_agents=n_agents,
                                         representation=representation,
                                         hidden_size=hidden_size,
                                         normalize=normalize,
                                         initialize=initialize,
                                         activation=activation,
                                         device=device,
                                         use_distributed_training=use_distributed_training,
                                         **kwargs)
        self.communicator = communicator
        self.config = kwargs['config']
        self.agent_keys = kwargs['agent_keys']

    @property
    def parameters_model(self):
        parameters_model = {}
        for key in self.model_keys:
            parameters_model[key] = list(self.representation[key].parameters()) + list(
                self.eval_Qhead[key].parameters()) + list(self.communicator[key].parameters())
        return parameters_model

    def forward(self, observation: Dict[str, Tensor], agent_ids: Tensor = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, alive_ally: dict = None):
        rnn_hidden_new, argmax_action, evalQ = {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        if self.config.use_parameter_sharing:
            key = self.model_keys[0]
            # [batch_size, n_agents, seq_length, 64]
            observation = self.communicator[key].obs_encode(observation[key])
            batch_size, seq_length = observation.shape[0], observation.shape[2]
            # [batch_size, seq_length, 64]
            gnn_input = {k: observation[:, i: i+1].squeeze(dim=1) for i, k in enumerate(self.agent_keys)}
            for i in range(self.config.convolution_layer):
                gnn_input = {k: self.communicator[key](k, gnn_input, alive_ally) for k in self.agent_keys}
            gnn_input = torch.stack([gnn_input[k] for k in self.agent_keys], dim=1)
            observation = {key: gnn_input.reshape(batch_size*self.n_agents, seq_length, -1)}
        else:
            observation = {k: self.communicator[k].obs_encode(observation[k]) for k in self.model_keys}
            for i in range(self.config.convolution_layer):
                observation = {k: self.communicator[k](k, observation, alive_ally) for k in self.model_keys}
        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}

        for key in agent_list:
            if self.use_rnn:
                outputs = self.representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
            else:
                q_inputs = outputs['state']

            evalQ[key] = self.eval_Qhead[key](q_inputs)

            if avail_actions is not None:
                evalQ_detach = evalQ[key].clone().detach()
                evalQ_detach[avail_actions[key] == 0] = -1e10
                argmax_action[key] = evalQ_detach.argmax(dim=-1, keepdim=False)
            else:
                argmax_action[key] = evalQ[key].argmax(dim=-1, keepdim=False)

        return rnn_hidden_new, argmax_action, evalQ

    def Qtarget(self, observation: Dict[str, Tensor], agent_ids: Dict[str, Tensor],
                agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, alive_ally: dict = None):
        rnn_hidden_new, q_target = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        if self.config.use_parameter_sharing:
            key = self.model_keys[0]
            # [batch_size, n_agents, seq_length, 64]
            observation = self.communicator[key].obs_encode(observation[key])
            batch_size, seq_length = observation.shape[0], observation.shape[2]
            # [batch_size, seq_length, 64]
            gnn_input = {k: observation[:, i: i+1].squeeze(dim=1) for i, k in enumerate(self.agent_keys)}
            for i in range(self.config.convolution_layer):
                gnn_input = {k: self.communicator[key](k, gnn_input, alive_ally) for k in self.agent_keys}
            observation = torch.stack([gnn_input[k] for k in self.agent_keys], dim=1)
            observation = {key: observation.reshape(batch_size*self.n_agents, seq_length, -1)}
        else:
            observation = {k: self.communicator[k].obs_encode(observation[k]) for k in self.model_keys}
            for i in range(self.config.convolution_layer):
                observation = {k: self.communicator[k](k, observation, alive_ally) for k in self.model_keys}
        for key in agent_list:
            if self.use_rnn:
                outputs = self.target_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.target_representation[key](observation[key])
                rnn_hidden_new[key] = None
            if self.use_parameter_sharing:
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
            else:
                q_inputs = outputs['state']
            q_target[key] = self.target_Qhead[key](q_inputs)
        return rnn_hidden_new, q_target