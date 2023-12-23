import numpy as np
import tensorflow.keras as tk
import tensorflow as tf
import torch
import torch_scatter


class DCG_utility(tk.Model):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super(DCG_utility, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        layers = [tk.layers.Dense(units=self.dim_hidden, activation='relu', input_shape=(self.dim_input,)),
                  tk.layers.Dense(units=self.dim_output, activation=None, input_shape=(self.dim_hidden,))]
        self.outputs = tk.Sequential(layers)

    def call(self, hidden_states_n, **kwargs):
        return self.outputs(hidden_states_n)


class DCG_payoff(DCG_utility):
    def __init__(self, dim_input, dim_hidden, dim_act, args):
        self.dim_act = dim_act
        self.low_rank_payoff = args.low_rank_payoff
        self.payoff_rank = args.payoff_rank
        dim_payoff_out = 2 * self.payoff_rank * self.dim_act if self.low_rank_payoff else self.dim_act ** 2
        super(DCG_payoff, self).__init__(dim_input, dim_hidden, dim_payoff_out)

    def call(self, hidden_states_n, edges_from=None, edges_to=None, **kwargs):
        input_payoff_0 = tf.concat([tf.gather(hidden_states_n, edges_from, axis=1),
                                    tf.gather(hidden_states_n, edges_to, axis=1)], axis=-1)
        input_payoff_1 = tf.concat([tf.gather(hidden_states_n, edges_to, axis=1),
                                    tf.gather(hidden_states_n, edges_from, axis=1)], axis=-1)
        input_payoff = tf.stack([input_payoff_0, input_payoff_1], axis=0)
        input_shape = input_payoff.shape
        payoffs = self.outputs(tf.reshape(input_payoff, [-1, input_shape[-1]]))
        payoffs = tf.reshape(payoffs, input_shape[:-1] + (self.dim_output, ))
        dim = payoffs.shape[0:-1]
        if self.low_rank_payoff:
            payoffs = payoffs.view(np.prod(dim) * self.payoff_rank, 2, self.dim_act)
            payoffs = tf.linalg.matmul(tf.expand_dims(payoffs[:, 0, :], -1),
                                       tf.expand_dims(payoffs[:, 1, :], -2))  # (dim_act * 1) * (1 * dim_act) -> (dim_act * dim_act)
            payoffs = tf.reduce_sum(tf.reshape(payoffs, list(dim) + [self.payoff_rank, self.dim_act, self.dim_act]), axis=-3)
        else:
            payoffs = tf.reshape(payoffs, list(dim) + [self.dim_act, self.dim_act])
        payoffs = tf.Variable(payoffs)
        payoffs[1].assign(tf.transpose(payoffs[1], perm=(0, 1, 3, 2)))  # f_ij(a_i, a_j) <-> f_ji(a_j, a_i)
        return tf.reduce_mean(payoffs, axis=0)  # f^E_{ij} = (f_ij(a_i, a_j) + f_ji(a_j, a_i)) / 2


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

    def set_coordination_graph(self):
        self.edges_from = torch.zeros(self.n_edges).long()
        self.edges_to = torch.zeros(self.n_edges).long()
        for i, edge in enumerate(self.edges):
            self.edges_from[i] = edge[0]
            self.edges_to[i] = edge[1]
        self.edges_n_in = torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                    index=self.edges_to, dim=0, dim_size=self.n_vertexes) \
                          + torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                      index=self.edges_from, dim=0, dim_size=self.n_vertexes)
        self.edges_n_in = self.edges_n_in.float()
        return
