from copy import deepcopy
from typing import Sequence, Optional, Union

import torch
from torch import nn
from torch.nn import Softmax

from xuance.torch.communications import IC3NetComm


class TarMAC(IC3NetComm):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 comm_passes: Optional[int] = 1,
                 model_keys: dict = None,
                 n_agents: int = 1,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(TarMAC, self).__init__(input_shape, hidden_sizes, comm_passes,
                                     model_keys, n_agents, device)

        self.config = kwargs['config']
        self.k_dim = self.config.dim_message
        self.v_dim = self.config.dim_message
        self.scale = self.k_dim ** -0.5
        self.use_gate = self.config.use_gate
        # query head for agent
        self.query_head = nn.Linear(self.recurrent_hidden_size, self.k_dim).to(self.device)
        # key head for other agents
        self.key_head = nn.Linear(self.recurrent_hidden_size, self.k_dim).to(self.device)
        # value head for other agents
        self.value_head = nn.Linear(self.recurrent_hidden_size, self.v_dim).to(self.device)


    def forward(self, key: str, obs: torch.Tensor, rnn_hidden: dict, alive_ally: dict) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # obs encode
        obs = self.obs_encoder(obs)
        alive_ally = {k: torch.as_tensor(alive_ally[k], dtype=torch.float32, device=self.device) for k in
                      self.model_keys}
        message = {
            key: deepcopy(value[0])
            for key, value in rnn_hidden.items()
        }
        # calculate the query for current agent
        q: torch.Tensor = self.query_head(obs)
        atten_scores, value_list = [], []
        for k in self.model_keys:
            if k != key:
                # calculate the signature/key for other agents
                message[k] = self.message_encode(message[k].transpose(0, 1))
                if self.use_gate:
                    prob = self.gate(message[k])
                    dist = torch.distributions.Categorical(prob)
                    value = self.value_head(message[k]) * alive_ally[k] * dist.sample().unsqueeze(dim=-1)
                else:
                    value = self.value_head(message[k]) * alive_ally[k]
                signature = self.key_head(message[k])
                dot = torch.matmul(signature, q.transpose(-1, -2)) * self.scale
                atten_scores.append(dot), value_list.append(value)

        atten_scores = torch.stack(atten_scores, dim=-1)
        value_list = torch.stack(value_list, dim=2)
        softmax_out = Softmax(dim=-1)(atten_scores)
        msg_receive = torch.matmul(softmax_out, value_list).squeeze(1)
        alive_agent_num = torch.sum(torch.stack(list(alive_ally.values()), dim=2), dim=2)
        alive_agent_num = torch.clamp(alive_agent_num, min=1.0)
        msg_receive = msg_receive / alive_agent_num
        return obs + msg_receive