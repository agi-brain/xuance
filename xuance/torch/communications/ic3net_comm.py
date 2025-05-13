from copy import deepcopy

import torch
import torch.nn as nn

from xuance.common import Optional, Union, Sequence
from xuance.torch import Module


class IC3NetComm(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 comm_passes: Optional[int] = 1,
                 model_keys: dict = None,
                 n_agents: int = 1,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__()

        self.input_shape = input_shape
        self.device = device
        self.fc_hidden_sizes = hidden_sizes["fc_hidden_sizes"]
        self.recurrent_hidden_size = hidden_sizes["recurrent_hidden_size"]
        self.comm_passes = comm_passes
        self.model_keys = model_keys
        self.n_agents = n_agents

        self.obs_encoder = nn.Linear(input_shape[0], self.recurrent_hidden_size).to(self.device)
        self.msg_encoder = nn.ModuleList([nn.Linear(self.recurrent_hidden_size, self.recurrent_hidden_size).to(self.device)
                                        for _ in range(self.comm_passes)])
        self.relu = nn.ReLU()

        self.gate = nn.Sequential(
            nn.Linear(self.recurrent_hidden_size, self.recurrent_hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(self.recurrent_hidden_size, 2),
            nn.Softmax(dim=-1)
        ).to(self.device)

    def message_encode(self, message: torch.Tensor) -> torch.Tensor:
        for i in range(self.comm_passes):
            message = self.msg_encoder[i](message)
            message = self.relu(message)
        return message

    def forward(self, key: str, obs: torch.Tensor, rnn_hidden: dict, alive_ally: dict) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # obs encode
        obs = self.obs_encoder(obs)
        msg_receive = torch.zeros_like(obs)
        alive_ally = {k: torch.as_tensor(alive_ally[k], dtype=torch.float32, device=self.device) for k in self.model_keys}
        message = {
            key: deepcopy(value[0])
            for key, value in rnn_hidden.items()
        }
        for k in self.model_keys:
            if k != key:
                message[k] = self.message_encode(message[k].transpose(0, 1))
                prob = self.gate(message[k])
                dist = torch.distributions.Categorical(prob)
                message[k] = message[k] * alive_ally[k]
                msg_receive = msg_receive + message[k] * dist.sample().unsqueeze(dim=-1)

        alive_agent_num = torch.sum(torch.stack(list(alive_ally.values()), dim=2), dim=2)
        alive_agent_num = torch.clamp(alive_agent_num, min=2.0)
        msg_receive = msg_receive / (alive_agent_num - 1)
        return obs + msg_receive