from copy import deepcopy
import torch
from xuance.torch import Tensor, ModuleDict
from xuance.torch.policies.deterministic_marl import MixingQnetwork
from xuance.common import Sequence, Optional, Callable, Union, Dict, List
from xuance.torch.policies import VDN_mixer, BasicQhead
from gymnasium.spaces import Discrete
from xuance.torch.utils import ModuleType


class ExpoCommQnetwork(MixingQnetwork):
    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation: ModuleDict,
                 mixer: Optional[VDN_mixer] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(ExpoCommQnetwork, self).__init__(action_space, n_agents, representation, mixer, hidden_size,
                                             normalize, initialize, activation, device, use_distributed_training,
                                             **kwargs)

        self.dim_input_Q, self.n_actions = {}, {}
        self.eval_Qhead, self.target_Qhead = ModuleDict(), ModuleDict()
        for key in self.model_keys:
            self.n_actions[key] = self.action_space[key].n
            self.dim_input_Q[key] = self.representation_info_shape[key]['state'][0] * 2
            if self.use_parameter_sharing:
                self.dim_input_Q[key] += self.n_agents
            self.eval_Qhead[key] = BasicQhead(self.dim_input_Q[key], self.n_actions[key], hidden_size,
                                              normalize, initialize, activation, device)
            self.target_Qhead[key] = deepcopy(self.eval_Qhead[key])
        self.communicator = kwargs['communicator']
        self.target_communicator = deepcopy(self.communicator)
        self.agent_keys = kwargs['agent_keys']

    @property
    def parameters_model(self):
        parameters_model = list(self.eval_Qtot.parameters()) + list(self.representation.parameters()) + list(
            self.eval_Qhead.parameters()) + list(self.communicator.parameters())
        return parameters_model
    
    def init_msg_prev(self, h):
        key = self.model_keys[0]
        self.communicator[key].msg_prev = torch.zeros_like(h[key][0])
        self.communicator[key].t = 0
    
    def init_msg_prev_target(self, h):
        key = self.model_keys[0]
        self.target_communicator[key].msg_prev = torch.zeros_like(h[key][0])
        self.target_communicator[key].t = 0

    def forward(self, observation: Dict[str, Tensor], agent_ids: Tensor = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, alive_ally = None):

        rnn_hidden_new, argmax_action, evalQ = {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}

        key = self.model_keys[0]
        seq_len = observation[key].shape[1]

        for key in agent_list:
            q_inputs = []

            # communication over time
            for t in range(seq_len):
                # [bs*self.n_agents, obs_dim]
                obs_i = observation[key][:, t:t+1]
                output = self.representation[key](obs_i, *rnn_hidden[key])
                agent_id_i = agent_ids[:, t].unsqueeze(1)
                h = output['state']
                alive_ally_i = {k: alive_ally[k][:, t:t + 1, :] for k in self.agent_keys}
                msg = self.communicator[key](h, alive_ally_i)

                q_inputs.append(torch.concat([output['state'], msg, agent_id_i], dim=-1))
                rnn_hidden[key] = (output['rnn_hidden'], output['rnn_cell'])
            
            rnn_hidden_new[key] = (rnn_hidden[key][0].detach(), rnn_hidden[key][1])
            
            q_inputs = torch.stack(q_inputs, dim=1).squeeze(2)

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
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, alive_ally = None):

        rnn_hidden_new, q_target = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        key = self.model_keys[0]
        seq_len = observation[key].shape[1]

        for key in agent_list:
            q_inputs = []
            # communication over time
            for t in range(seq_len):
                # [bs*self.n_agents, obs_dim]
                obs_i = observation[key][:, t:t+1]
                output = self.target_representation[key](obs_i, *rnn_hidden[key])
                agent_id_i = agent_ids[:, t].unsqueeze(1)
                h = output['state']
                alive_ally_i = {k: alive_ally[k][:, t:t + 1, :] for k in self.agent_keys}
                msg = self.target_communicator[key](h, alive_ally_i)

                q_inputs.append(torch.concat([output['state'], msg, agent_id_i], dim=-1))
                rnn_hidden[key] = (output['rnn_hidden'], output['rnn_cell'])
            
            rnn_hidden_new[key] = (rnn_hidden[key][0].detach(), rnn_hidden[key][1])
            
            q_inputs = torch.stack(q_inputs, dim=1).squeeze(2)
            q_target[key] = self.target_Qhead[key](q_inputs)
        return rnn_hidden_new, q_target


    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qtot.parameters(), self.target_Qtot.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.communicator.parameters(), self.target_communicator.parameters()):
            tp.data.copy_(ep)