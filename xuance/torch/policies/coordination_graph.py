import torch
import torch.nn as nn
import numpy as np
from xuance.common import Optional, Union
from xuance.torch import Tensor
try:
    import torch_scatter
except ImportError:
    print("The module torch_scatter is not installed.")


class DCG_utility(nn.Module):
    """
    The utility module for deep coordination graph.

    Args:
        dim_input (int): The dimension of input for the utility module.
        dim_hidden (int): The dimension of hidden layer for the utility module.
        dim_output (int): The dimension of output for the utility module.
        device (Optional[Union[str, int, torch.device]]): The device for running the model, default is None.
    """

    def __init__(self,
                 dim_input: int,
                 dim_hidden: int,
                 dim_output: int,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(DCG_utility, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.device = device
        '''All utilities share the same parameters'''
        self.output = nn.Sequential(nn.Linear(self.dim_input, self.dim_hidden),
                                    nn.ReLU(),
                                    nn.Linear(self.dim_hidden, self.dim_output)).to(device)

    def forward(self, hidden_states_n: Tensor):
        """
        Calculate the utility values for multiple agents.

        Args:
            hidden_states_n (Tensor): The hidden states for the representations of n agents.

        Returns: The utility values for multiple agents.
        """
        return self.output(hidden_states_n)


class DCG_payoff(DCG_utility):
    """
    The payoff module for deep coordination graph.

    Args:
        dim_input (int): The dimension of input for the payoff module.
        dim_hidden (int): The dimension of hidden layer for the payoff module.
        dim_act (int): The dimension of actions.
        low_rank_payoff (int): The low rank payoff.
        payoff_rank (int): The rank of payoff.
        device (Optional[Union[str, int, torch.device]]): The device for running the model, default is None.
    """
    def __init__(self,
                 dim_input: int,
                 dim_hidden: int,
                 dim_act: int,
                 low_rank_payoff: int,
                 payoff_rank: int,
                 device: Optional[Union[str, int, torch.device]] = None):
        self.dim_act = dim_act
        self.low_rank_payoff = low_rank_payoff
        self.payoff_rank = payoff_rank
        dim_payoff_out = 2 * self.payoff_rank * self.dim_act if self.low_rank_payoff else self.dim_act ** 2
        super(DCG_payoff, self).__init__(dim_input, dim_hidden, dim_payoff_out, device)

    def forward(self, hidden_states_n: Tensor,
                edges_from: Tensor=None,
                edges_to: Tensor=None):
        """
        Calculate the payoff values for the graph constructed by multiple agents.

        Args:
            hidden_states_n: The hidden states for the representations of n agents.
            edges_from: The edges from others to self, default is None.
            edges_to: The edges from self to others, default is None.

        Returns: Mean of payoff values for edge_from and edge_to.
        """
        input_payoff = torch.stack([torch.cat([hidden_states_n[:, edges_from], hidden_states_n[:, edges_to]], dim=-1),
                                    torch.cat([hidden_states_n[:, edges_to], hidden_states_n[:, edges_from]], dim=-1)],
                                   dim=0)
        payoffs = self.output(input_payoff)
        dim = payoffs.shape[0:-1]
        if self.low_rank_payoff:
            payoffs = payoffs.view(np.prod(dim) * self.payoff_rank, 2, self.dim_act)
            payoffs = torch.matmul(payoffs[:, 0, :].unsqueeze(dim=-1), payoffs[:, 1, :].unsqueeze(
                dim=-2))  # (dim_act * 1) * (1 * dim_act) -> (dim_act * dim_act)
            payoffs = payoffs.view(list(dim) + [self.payoff_rank, self.dim_act, self.dim_act]).sum(dim=-3)
        else:
            payoffs = payoffs.view(list(dim) + [self.dim_act, self.dim_act])
        payoffs[1] = payoffs[1].transpose(dim0=-1, dim1=-2).clone()  # f_ij(a_i, a_j) <-> f_ji(a_j, a_i)
        return payoffs.mean(dim=0)  # f^E_{ij} = (f_ij(a_i, a_j) + f_ji(a_j, a_i)) / 2


class Coordination_Graph(object):
    """
    Construct a deep coordination graph.

    Args:
        n_vertexes (int): The number of vertexes in the graph.
        graph_type (str): The type of graph, default is "FULL".
    """
    def __init__(self,
                 n_vertexes: int,
                 graph_type: str = "FULL",
                 device: Optional[Union[str, int, torch.device]] = None):
        self.n_vertexes = n_vertexes
        self.device = device
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

    def set_coordination_graph(self):
        """ Reset the coordination graph. """
        self.edges_from = torch.zeros(self.n_edges).long().to(self.device)
        self.edges_to = torch.zeros(self.n_edges).long().to(self.device)
        for i, edge in enumerate(self.edges):
            self.edges_from[i] = edge[0]
            self.edges_to[i] = edge[1]
        self.edges_n_in = torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                    index=self.edges_to, dim=0, dim_size=self.n_vertexes) \
                          + torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                      index=self.edges_from, dim=0, dim_size=self.n_vertexes)
        self.edges_n_in = self.edges_n_in.float()
        return
