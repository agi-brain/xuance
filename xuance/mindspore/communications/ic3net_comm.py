from argparse import Namespace
from typing import Sequence, Optional, Union

import torch

from xuance.torch.communications.comm_net import CommNet


class IC3NetComm(CommNet):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 comm_passes: Optional[int] = 1,
                 model_keys: dict = None,
                 agent_keys: dict = None,
                 n_agents: int = 1,
                 device: Optional[Union[str, int, torch.device]] = None,
                 config: Optional[Namespace] = None,
                 **kwargs):
        super(IC3NetComm, self).__init__(input_shape, hidden_sizes, comm_passes, model_keys,
                                          agent_keys, n_agents, device, config, **kwargs)

    def forward(self, obs: torch.Tensor, msg_send: dict, alive_ally: dict, gate_control: dict = None,):
        alive_ally = {k: torch.as_tensor(alive_ally[k], dtype=torch.float32, device=self.device) for k in
                      self.agent_keys}
        batch_size, seq_length = obs.shape[0], obs.shape[1]
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            msg_send = msg_send[key].view(batch_size // self.n_agents, self.n_agents, seq_length, -1)
            alive_ally = torch.stack(list(alive_ally.values()), dim=1)
            gate_control = gate_control[key].view(batch_size // self.n_agents, self.n_agents, -1)
            msg_send = msg_send * alive_ally * gate_control.unsqueeze(-1)
            message = torch.sum(msg_send, dim=1, keepdim=True) - msg_send
            alive_agent_num = torch.sum(alive_ally, dim=1).unsqueeze(1)
            alive_agent_num = torch.clamp(alive_agent_num, min=1.0)
        else:
            message = {k: msg_send[k] * alive_ally[k] * gate_control[k].unsqueeze(-1) for k in self.model_keys}
            alive_ally = torch.stack(list(alive_ally.values()), dim=1)
            message = torch.stack(list(message.values()), dim=0)
            message = torch.sum(message, dim=0)
            alive_agent_num = torch.sum(alive_ally, dim=1)
            alive_agent_num = torch.clamp(alive_agent_num, min=1.0)
        message = message / alive_agent_num
        msg_receive = self.message_encode(message)
        if self.use_parameter_sharing:
            msg_receive = msg_receive.view(batch_size, seq_length, -1)
        return msg_receive


