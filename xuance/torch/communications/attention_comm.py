from copy import deepcopy
from typing import Sequence, Optional, Union

import torch
from torch import nn
from xuance.torch.communications import IC3NetComm


class TarMAC(IC3NetComm):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 comm_passes: Optional[int] = 1,
                 model_keys: dict = None,
                 agent_keys: dict = None,
                 n_agents: int = 1,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(TarMAC, self).__init__(input_shape, hidden_sizes, comm_passes, model_keys,
                                          agent_keys, n_agents, device, **kwargs)

    def forward(self, obs: torch.Tensor, rnn_hidden: dict, alive_ally: dict) -> torch.Tensor:
        alive_ally = {k: torch.as_tensor(alive_ally[k], dtype=torch.float32, device=self.device) for k in
                      self.agent_keys}
        batch_size, seq_length = obs.shape[0], obs.shape[1]
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            rnn_input = deepcopy(rnn_hidden)
            rnn_input = {key: [rnn_input[key][0].view(seq_length, batch_size // self.n_agents, self.n_agents, -1),
                               rnn_input[key][1].view(seq_length, batch_size // self.n_agents, self.n_agents, -1)]}
            alive_ally = {key: torch.stack(list(alive_ally.values()), dim=0)}
            rnn_input = rnn_input[key][0].permute(2, 1, 0, 3)
            gate_control = self.gate(rnn_input) > 0.5
            message = [rnn_input * alive_ally[key] * gate_control]
            alive_agent_num = torch.sum(alive_ally[key], dim=0).unsqueeze(0)
            alive_agent_num = torch.clamp(alive_agent_num, min=1.0)
        else:
            rnn_input = deepcopy(rnn_hidden)
            gate_control = {k: self.gate(rnn_input[k][0].transpose(0, 1)) > 0.5 for k in self.agent_keys}
            message = [rnn_input[k][0].transpose(0, 1) * alive_ally[k] * gate_control[k] for k in self.model_keys]
            alive_agent_num = torch.sum(torch.stack(list(alive_ally.values()), dim=2), dim=2)
            alive_agent_num = torch.clamp(alive_agent_num, min=1.0)
        message = torch.stack(message, dim=0)
        message = torch.sum(message, dim=0).squeeze(dim=0)
        message = message / alive_agent_num
        msg_receive = self.message_encode(message)
        if self.use_parameter_sharing:
            msg_receive = msg_receive.view(batch_size, seq_length, -1)
        return obs + msg_receive