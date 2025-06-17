from copy import deepcopy

import torch
import torch.nn as nn

from xuance.common import Optional, Union, Sequence
from xuance.torch import Module


class CommNet(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 comm_passes: Optional[int] = 1,
                 model_keys: dict = None,
                 agent_keys: dict = None,
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
        self.agent_keys = agent_keys
        self.n_agents = n_agents

        self.obs_encoder = nn.Sequential(nn.Linear(input_shape[0], self.recurrent_hidden_size * 2),
                                         nn.ReLU(),
                                         nn.Linear(self.recurrent_hidden_size * 2, self.recurrent_hidden_size)).to(self.device)
        self.msg_encoder = nn.Linear(self.recurrent_hidden_size, self.recurrent_hidden_size).to(self.device)
        self.use_parameter_sharing = kwargs['use_parameter_sharing']

    def message_encode(self, message: torch.Tensor) -> torch.Tensor:
        return self.msg_encoder(message)

    def obs_encode(self, observation):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        return self.obs_encoder(obs)

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
            message = [rnn_input * alive_ally[key]]
            alive_agent_num = torch.sum(alive_ally[key], dim=0).unsqueeze(0)
            alive_agent_num = torch.clamp(alive_agent_num, min=1.0)
        else:
            rnn_input = deepcopy(rnn_hidden)
            message = [rnn_input[k][0].transpose(0, 1) * alive_ally[k] for k in self.model_keys]
            alive_agent_num = torch.sum(torch.stack(list(alive_ally.values()), dim=2), dim=2)
            alive_agent_num = torch.clamp(alive_agent_num, min=1.0)
        message = torch.stack(message, dim=0)
        message = torch.sum(message, dim=0).squeeze(dim=0)
        message = message / alive_agent_num
        msg_receive = self.message_encode(message)
        if self.use_parameter_sharing:
            msg_receive = msg_receive.view(batch_size, seq_length, -1)
        return obs + msg_receive