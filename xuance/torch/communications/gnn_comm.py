from argparse import Namespace
from typing import Optional, Sequence, Union
import torch
from torch import nn
from xuance.torch.communications.comm_net import CommNet


class DGNComm(CommNet):
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
        super(DGNComm, self).__init__(input_shape, hidden_sizes, comm_passes, model_keys, agent_keys, n_agents, device,
                                      config, **kwargs)

        self.n_head = config.n_head
        self.convolution_layers = config.convolution_layers
        self.gcn = self.build_gcn(self.convolution_layers, self.n_head)
        self.activation = nn.ReLU()

    def build_gcn(self, num_layers, num_heads, activation = nn.ReLU()):
        gcn_layers = []
        for _ in range(num_layers):
            gcn_layers.append(GraphMultiHeadAttentionLayer(self.recurrent_hidden_size, self.recurrent_hidden_size, num_heads))
            gcn_layers.append(activation)
        return nn.Sequential(*gcn_layers)

    def create_adjacency_matrix(self, alive_ally):
        adj_matrix = torch.matmul(alive_ally, alive_ally.transpose(1, 2))
        return adj_matrix

    def gcn_block(self, x, adj_matrix):
        return self.gcn(x, adj_matrix)

    def forward(self, obs: torch.Tensor, msg_send: dict, alive_ally: dict) -> torch.Tensor:
        alive_ally = {k: torch.as_tensor(alive_ally[k], dtype=torch.float32, device=self.device) for k in
                      self.agent_keys}
        batch_size, seq_length = obs.shape[0], obs.shape[1]
        key = self.model_keys[0]
        msg_send = msg_send[key].view(batch_size // self.n_agents, self.n_agents, seq_length, -1)
        alive_ally = torch.stack(list(alive_ally.values()), dim=1)
        alive_ally = alive_ally.squeeze(2)
        msg_send = msg_send.squeeze(2)
        adj_matrix = self.create_adjacency_matrix(alive_ally)
        message = self.gcn_block(msg_send, adj_matrix)
        msg_receive = self.msg_encoder(message)
        msg_receive = msg_receive.view(batch_size, seq_length, -1)
        return msg_receive


class GraphMultiHeadAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=2, dropout=0.6, concat=True, device='cpu'):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.concat = concat

        self.head_dim = output_dim // num_heads
        assert self.head_dim * num_heads == output_dim, "output_dim必须能被num_heads整除"

        self.W = nn.Linear(input_dim, num_heads * self.head_dim).to(self.device)
        self.a = nn.Parameter(torch.empty(size=(2 * self.head_dim, 1))).to(self.device)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, adj):
        """
        x: node feature matrix [batch_size, n_agents, input_dim]
        adj: adjacency matrix [batch_size, n_agents, n_agents]
        """
        batch_size, n_agents, _ = x.shape

        # [batch_size, n_agents, num_heads * head_dim]
        h = self.W(x)
        h = h.view(batch_size, n_agents, self.num_heads, self.head_dim)

        # h_i: [batch_size, n_agents, num_heads, head_dim]
        # h_j: [batch_size, n_agents, num_heads, head_dim]
        h_i = h.unsqueeze(2)  # [batch_size, n_agents, 1, num_heads, head_dim]
        h_j = h.unsqueeze(1)  # [batch_size, 1, n_agents, num_heads, head_dim]

        # [batch_size, n_agents, n_agents, num_heads, 2*head_dim]
        h_cat = torch.cat([h_i.repeat(1, 1, n_agents, 1, 1),
                           h_j.repeat(1, n_agents, 1, 1, 1)], dim=-1)

        # [batch_size, n_agents, n_agents, num_heads]
        e = self.leakyrelu(torch.matmul(h_cat, self.a).squeeze(-1))

        mask = -9e15 * torch.ones_like(e)
        adj_expanded = adj.unsqueeze(-1).repeat(1, 1, 1, self.num_heads)
        e = torch.where(adj_expanded > 0, e, mask)

        attention = nn.Softmax(dim=2)(e)
        attention = self.dropout(attention)

        # attention: [batch_size, n_agents, n_agents, num_heads]
        # h: [batch_size, n_agents, num_heads, head_dim]
        h = h.permute(0, 2, 1, 3)  # [batch_size, num_heads, n_agents, head_dim]

        output = torch.zeros(batch_size, n_agents, self.num_heads, self.head_dim).to(self.device)

        for head in range(self.num_heads):
            att_head = attention[:, :, :, head]  # [batch_size, n_agents, n_agents]
            h_head = h[:, head, :, :]  # [batch_size, n_agents, head_dim]
            output[:, :, head, :] = torch.bmm(att_head, h_head)

        if self.concat:
            output = output.reshape(batch_size, n_agents, self.output_dim)
        else:
            output = output.mean(dim=2)

        return output