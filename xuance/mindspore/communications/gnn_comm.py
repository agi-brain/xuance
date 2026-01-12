from typing import Sequence, Optional, Union

import torch
from torch import nn


class DGNComm(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 atten_head: Optional[int] = 1,
                 agent_keys: dict = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs
                ):
        super(DGNComm, self).__init__()
        self.input_shape = input_shape
        self.device = device
        self.fc_hidden_sizes = hidden_sizes["fc_hidden_sizes"]
        self.recurrent_hidden_size : int = hidden_sizes["recurrent_hidden_size"]
        self.agent_keys = agent_keys
        self.config = kwargs['config']

        self.obs_encoder = nn.Linear(input_shape[0], self.recurrent_hidden_size, device=self.device)
        self.atten_head = atten_head
        self.q_dim = self.recurrent_hidden_size // self.atten_head
        self.scale = self.q_dim ** -0.5
        self.q = nn.ModuleList(
            nn.Linear(self.recurrent_hidden_size, self.q_dim).to(self.device) for i in range(self.atten_head))
        self.k = nn.ModuleList(
            nn.Linear(self.recurrent_hidden_size, self.q_dim).to(self.device) for i in range(self.atten_head))
        self.v = nn.ModuleList(
            nn.Linear(self.recurrent_hidden_size, self.q_dim).to(self.device) for i in range(self.atten_head))

    def obs_encode(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        return self.obs_encoder(observation)

    def gcn(self, obs, matrix, alive_ally):
        global atten_scores
        alive_agent_num = torch.sum(torch.stack(list(alive_ally.values()), dim=2), dim=2)
        alive_agent_num = torch.clamp(alive_agent_num, min=2.0)
        matrix = [data / alive_agent_num for data in matrix]
        matrix = torch.stack(matrix, dim=-2)
        gnn_out = []
        for i in range(self.atten_head):
            atten_query = self.q[i](obs).unsqueeze(dim=-2)
            atten_key = self.k[i](matrix)
            atten_scores = nn.Softmax(dim=-1)(torch.matmul(atten_query, atten_key.transpose(-1, -2))) * self.scale
            atten_value = self.v[i](matrix)
            gnn_out.append(torch.matmul(atten_scores, atten_value).squeeze(-2))
        gnn_out = torch.cat(gnn_out, dim=-1)
        return obs + gnn_out


    def forward(self, key: str, obs: dict, alive_ally: dict):
        alive_ally = {k: torch.as_tensor(alive_ally[k], dtype=torch.float32, device=self.device) for k in
                      alive_ally.keys()}
        # get matrix
        matrix = []
        matrix.append(obs[key])
        for k in self.agent_keys:
            if k != key:
                matrix.append(obs[k] * alive_ally[k])
        gcn_out = self.gcn(obs[key], matrix, alive_ally)
        return gcn_out
