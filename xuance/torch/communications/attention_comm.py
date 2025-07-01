from argparse import Namespace
from copy import deepcopy
from typing import Sequence, Optional, Union
import torch
from torch import nn
from xuance.torch.communications.ic3net_comm import IC3NetComm


class TarMAC(IC3NetComm):
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
        super(TarMAC, self).__init__(input_shape, hidden_sizes, comm_passes, model_keys,
                                     agent_keys, n_agents, device, config, **kwargs)
        self.q = nn.Linear(self.recurrent_hidden_size, self.recurrent_hidden_size).to(self.device)
        self.k = nn.Linear(self.recurrent_hidden_size, self.recurrent_hidden_size).to(self.device)
        self.v = nn.Linear(self.recurrent_hidden_size, self.recurrent_hidden_size).to(self.device)
        self.scale = self.recurrent_hidden_size ** 0.5

    def forward(self, obs: torch.Tensor, msg_send: dict, alive_ally: dict, gate_control: dict = None,):
        alive_ally = {k: torch.as_tensor(alive_ally[k], dtype=torch.float32, device=self.device) for k in
                      self.agent_keys}
        batch_size, seq_length = obs.shape[0], obs.shape[1]
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            msg_send = msg_send[key].view(batch_size // self.n_agents, self.n_agents, seq_length, -1)
            alive_ally = torch.stack(list(alive_ally.values()), dim=1)
            gate_control = gate_control[key].view(batch_size // self.n_agents, self.n_agents, -1)
            msg_send = msg_send * alive_ally
            if self.config.use_gate:
                msg_send = msg_send * gate_control.unsqueeze(-1)
            q, k, v = self.q(msg_send), self.k(msg_send), self.v(msg_send)
            attn_scores = torch.einsum('bikd,bjkd->bikj', q, k) / self.scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            message = torch.einsum('bikj,bjkd->bikd', attn_weights, v)
            message = message.view(batch_size, seq_length, -1)
        else:
            message = {k: msg_send[k] * alive_ally[k] * gate_control[k].unsqueeze(-1) for k in self.model_keys}
            message = torch.stack(list(message.values()), dim=0)
            message = torch.sum(message, dim=0)
        msg_receive = self.message_encode(message)
        return msg_receive