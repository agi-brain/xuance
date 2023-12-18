import torch
import torch.nn as nn
import numpy as np
import torch_scatter


class DCG_utility(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super(DCG_utility, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.output = nn.Sequential(nn.Linear(self.dim_input, self.dim_hidden),
                                    nn.ReLU(),
                                    nn.Linear(self.dim_hidden, self.dim_output))
        # self.output = nn.Sequential(nn.Linear(self.dim_input, self.dim_output))

    def forward(self, hidden_states_n):
        return self.output(hidden_states_n)


class DCG_payoff(DCG_utility):
    def __init__(self, dim_input, dim_hidden, dim_act, args):
        self.dim_act = dim_act
        self.low_rank_payoff = args.low_rank_payoff
        self.payoff_rank = args.payoff_rank
        dim_payoff_out = 2 * self.payoff_rank * self.dim_act if self.low_rank_payoff else self.dim_act ** 2
        super(DCG_payoff, self).__init__(dim_input, dim_hidden, dim_payoff_out)

    def forward(self, hidden_states_n, edges_from=None, edges_to=None):
        input_payoff = torch.stack([torch.cat([hidden_states_n[:, edges_from], hidden_states_n[:, edges_to]], dim=-1),
                                    torch.cat([hidden_states_n[:, edges_to], hidden_states_n[:, edges_from]], dim=-1)],
                                   dim=0)
        payoffs = self.output(input_payoff)
        dim = payoffs.shape[0:-1]
        if self.low_rank_payoff:
            payoffs = payoffs.view(np.prod(dim)*self.payoff_rank, 2, self.dim_act)
            payoffs = torch.matmul(payoffs[:, 0, :].unsqueeze(dim=-1), payoffs[:, 1, :].unsqueeze(dim=-2))  # (dim_act * 1) * (1 * dim_act) -> (dim_act * dim_act)
            payoffs = payoffs.view(list(dim)+[self.payoff_rank, self.dim_act, self.dim_act]).sum(dim=-3)
        else:
            payoffs = payoffs.view(list(dim)+[self.dim_act, self.dim_act])
        payoffs[1] = payoffs[1].transpose(dim0=-1, dim1=-2).clone()  # f_ij(a_i, a_j) <-> f_ji(a_j, a_i)
        return payoffs.mean(dim=0)  # f^E_{ij} = (f_ij(a_i, a_j) + f_ji(a_j, a_i)) / 2


class Coordination_Graph(object):
    def __init__(self, n_vertexes, graph_type):
        self.n_vertexes = n_vertexes
        self.edges = []
        if graph_type == "CYCLE":
            self.edges = [(i, i + 1) for i in range(self.n_vertexes - 1)] + [(self.n_vertexes - 1, 0)]
        elif graph_type == "LINE":
            self.edges = [(i, i + 1) for i in range(self.n_vertexes - 1)]
        elif graph_type == "STAR":
            self.edges = [(0, i + 1) for i in range(self.n_vertexes - 1)]
        elif graph_type == "VDN":
            pass
        elif graph_type == "FULL":
            self.edges = [[(j, i + j + 1) for i in range(self.n_vertexes - j - 1)] for j in range(self.n_vertexes - 1)]
            self.edges = [e for l in self.edges for e in l]
        else:
            raise AttributeError("There is no graph type named {}!".format(graph_type))
        self.n_edges = len(self.edges)
        self.edges_from = None
        self.edges_to = None

    def set_coordination_graph(self, device):
        self.edges_from = torch.zeros(self.n_edges).long().to(device)
        self.edges_to = torch.zeros(self.n_edges).long().to(device)
        for i, edge in enumerate(self.edges):
            self.edges_from[i] = edge[0]
            self.edges_to[i] = edge[1]
        self.edges_n_in = torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                    index=self.edges_to, dim=0, dim_size=self.n_vertexes) \
                          + torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                      index=self.edges_from, dim=0, dim_size=self.n_vertexes)
        self.edges_n_in = self.edges_n_in.float()
        return
