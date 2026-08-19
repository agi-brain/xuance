from argparse import Namespace
from copy import deepcopy

import torch
import torch.nn as nn
from xuance.common import Optional, Union, Sequence
from xuance.torch import Module
import torch.nn.functional as F


class ExpoComm(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 state_shape: Sequence[int],
                 comm_passes: Optional[int] = 1,
                 model_keys: dict = None,
                 agent_keys: dict = None,
                 n_agents: int = 1,
                 device: Optional[Union[str, int, torch.device]] = None,
                 config: Optional[Namespace] = None,
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
        self.config = config
        self.use_parameter_sharing = self.config.use_parameter_sharing
        self.topk = self.config.topk
        self.attention_dim = self.config.attention_dim
        self.one_peer = self.config.one_peer
        self.state_dim = state_shape[0]
        
        self.obs_encoder = nn.Linear(self.input_shape[0], self.recurrent_hidden_size, device=self.device)

        self.msg_prev = None
        self.t = 0

        # message aggregation for static
        self.msg_q = nn.Linear(self.recurrent_hidden_size, self.attention_dim, device=self.device)
        self.msg_k = nn.Linear(self.recurrent_hidden_size, self.attention_dim, device=self.device)
        self.msg_v = nn.Linear(self.recurrent_hidden_size, self.recurrent_hidden_size, device=self.device)
        
        # message aggregation for one peer
        self.msg_processor = nn.Linear(self.recurrent_hidden_size * 2, self.recurrent_hidden_size, device=self.device)
        self.msg_rnn = nn.GRUCell(self.recurrent_hidden_size, self.recurrent_hidden_size, device=self.device)

        # predict net for aux loss
        self.predict_net = nn.Sequential(
            nn.Linear(self.recurrent_hidden_size, self.recurrent_hidden_size),
            nn.ReLU(),
            nn.Linear(self.recurrent_hidden_size, self.state_dim),
        ).to(self.device)

    def obs_encode(self, observation):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        return self.obs_encoder(obs)

    def get_exp_neighbors(self, bs, n_agents, topk):
        """
        positions: (batch_size, n_agents* 2)
        """
        topk_indices = torch.arange(topk - 1)
        topk_indices = torch.pow(2, topk_indices)
        topk_indices = torch.cat([torch.zeros(1), topk_indices])

        agent_ind = torch.arange(n_agents)
        topk_indices = agent_ind[:, None] + topk_indices[None, :]
        topk_indices = topk_indices % n_agents
        topk_indices = topk_indices[None, :, :].expand(bs, -1, -1)

        if self.one_peer:
            topk_index = self.t % self.topk
            topk_indices = topk_indices[:, :, topk_index]
            self.t += 1

        topk_indices = topk_indices.long().to(self.device)

        # (bs, n_agents, topk)
        return topk_indices
    
    def forward(self, h, alive_ally):
        """
            h: agents hidden state
            other_msg: other agents message at last step
            alive_ally: alive mask of agents
        """
        alive_ally = {k: torch.as_tensor(alive_ally[k], dtype=torch.float32, device=self.device) for k in
                      self.agent_keys}

        alive_mask = torch.stack(list(alive_ally.values()), dim=1).squeeze(-1).squeeze(-1)
        bs = h.shape[0] // self.n_agents

        topk_indices = self.get_exp_neighbors(bs, self.n_agents, self.topk)

        if self.one_peer:
            topk_indices = topk_indices[:, :, None, None].expand(
            -1, -1, -1, self.recurrent_hidden_size
            )

            msg_ego = self.msg_prev.reshape(bs*self.n_agents, -1)

            other_msg = self.msg_prev.reshape(bs, 1, self.n_agents, self.recurrent_hidden_size).expand(
                -1, self.n_agents, -1, -1
            )

            msg_received = other_msg.gather(dim=2, index=topk_indices)
            msg_received = msg_received[:, :, 0, :]
            ego_h = h.reshape(bs, self.n_agents, -1)

            msg_input = self.msg_processor(torch.cat([ego_h, msg_received], dim=-1))
            msg_input = msg_input.reshape(bs*self.n_agents, -1)
            m_aggregated = self.msg_rnn(msg_input, msg_ego)


        else:
            topk_indices = topk_indices[:, :, :, None].expand(
            -1, -1, -1, self.recurrent_hidden_size
            )

            other_msg = self.msg_prev.reshape(bs, 1, self.n_agents, self.recurrent_hidden_size).expand(
                -1, self.n_agents, -1, -1
            )

            msg_received = other_msg.gather(dim=2, index=topk_indices)

            ego_h = h.reshape(bs, self.n_agents, -1)

            q = self.msg_q(ego_h).reshape(bs, self.n_agents, self.attention_dim, 1)
            k = self.msg_k(msg_received)
            attention = torch.matmul(k, q)[:, :, :, 0]
            attention = F.softmax(attention / torch.sqrt(torch.tensor(self.attention_dim, device=self.device)),
                                    dim=-1)
            m = self.msg_v(msg_received)
            m_aggregated = (attention[:, :, :, None] * m).sum(dim=2).reshape(bs * self.n_agents, 1, -1)

        self.msg_prev = m_aggregated

        return m_aggregated

