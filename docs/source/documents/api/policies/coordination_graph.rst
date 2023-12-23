Coordination-Graph
==============================================

A key part for deep coordination graph-based algorithms, 
where utility functions and payoffs are computed based on the interactions between different agents.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.policies.coordination_graph.DCG_utility(dim_input, dim_hidden, dim_output)

  Defines a sequential neural network (output) with linear layers and ReLU activation.
  It represents the utility function.

  :param dim_input: The dimension of the input.
  :type dim_input: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param dim_output: The dimension of the output.
  :type dim_output: int

.. py:function::
  xuance.torch.policies.coordination_graph.DCG_utility.forward(hidden_states_n)

  :param hidden_states_n: The dimension of the hidden states of n agents.
  :type hidden_states_n: int
  :return: The utility values for agents.
  :rtype: Tensor

.. py:class::
  xuance.torch.policies.coordination_graph.DCG_payoff(dim_input, dim_hidden, dim_act, args)

  This class defines the payoff function between agents, which is inherits from DCG_utility.

  :param dim_input: The dimension of the input.
  :type dim_input: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param dim_act: The dimension of the actions.
  :type dim_act: int
  :param args: the arguments.
  :type args: Namespace

.. py:function::
  xuance.torch.policies.coordination_graph.DCG_payoff.forward(hidden_states_n, edges_from, edges_to)

  Computes payoffs based on the provided hidden states and edge information.
  Supports both low-rank and full-rank payoff computation.

  :param hidden_states_n: The dimension of the hidden states of n agents.
  :type hidden_states_n: int
  :param edges_from: The edges from others to self agent.
  :type edges_from: Tensor
  :param edges_to: The edges from  self agent to others.
  :type edges_to: Tensor
  :return: The provided hidden states and edge information.
  :rtype: Tensor

.. py:class::
  xuance.torch.policies.coordination_graph.Coordination_Graph(n_vertexes, graph_type)

  Represents a coordination graph with a specified number of vertices and type.

  :param n_vertexes: The number of vertexes between agents.
  :type n_vertexes: int
  :param graph_type: The type of the topology graph for n agents.
  :type graph_type: str

.. py:function::
  xuance.torch.policies.coordination_graph.Coordination_Graph.set_coordination_graph(device)

  Sets up the coordination graph, including the assignment of edges and related tensors.

  :param device: The calculating device.
  :type device: str

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.policies.coordination_graph.DCG_utility(dim_input, dim_hidden, dim_output)

  Defines a sequential neural network (output) with linear layers and ReLU activation. 
  It represents the utility function.

  :param dim_input: The dimension of the input.
  :type dim_input: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param dim_output: The dimension of the output.
  :type dim_output: int

.. py:function::
  xuance.tensorflow.policies.coordination_graph.DCG_utility.call(hidden_states_n)

  :param hidden_states_n: The dimension of the hidden states of n agents.
  :type hidden_states_n: int
  :return: The utility values for agents.
  :rtype: Tensor

.. py:class::
  xuance.tensorflow.policies.coordination_graph.DCG_payoff(dim_input, dim_hidden, dim_act, args)

  This class defines the payoff function between agents, which is inherits from DCG_utility.

  :param dim_input: The dimension of the input.
  :type dim_input: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param dim_act: The dimension of the actions.
  :type dim_act: int
  :param args: the arguments.
  :type args: Namespace

.. py:function::
  xuance.tensorflow.policies.coordination_graph.DCG_payoff.call(hidden_states_n, edges_from, edges_to)

  Computes payoffs based on the provided hidden states and edge information. 
  Supports both low-rank and full-rank payoff computation.

  :param hidden_states_n: The dimension of the hidden states of n agents.
  :type hidden_states_n: int
  :param edges_from: The edges from others to self agent.
  :type edges_from: Tensor
  :param edges_to: The edges from  self agent to others.
  :type edges_to: Tensor
  :return: The provided hidden states and edge information.
  :rtype: Tensor

.. py:class::
  xuance.tensorflow.policies.coordination_graph.Coordination_Graph(n_vertexes, graph_type)

  Represents a coordination graph with a specified number of vertices and type.

  :param n_vertexes: The number of vertexes between agents.
  :type n_vertexes: int
  :param graph_type: The type of the topology graph for n agents.
  :type graph_type: str

.. py:function::
  xuance.tensorflow.policies.coordination_graph.Coordination_Graph.set_coordination_graph()

  Sets up the coordination graph, including the assignment of edges and related tensors.

.. raw:: html

    <br><hr>


MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.policies.coordination_graph.DCG_utility(dim_input, dim_hidden, dim_output)

  Defines a sequential neural network (output) with linear layers and ReLU activation. 
  It represents the utility function.

  :param dim_input: The dimension of the input.
  :type dim_input: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param dim_output: The dimension of the output.
  :type dim_output: int

.. py:function::
  xuance.mindspore.policies.coordination_graph.DCG_utility.call(hidden_states_n)

  :param hidden_states_n: The dimension of the hidden states of n agents.
  :type hidden_states_n: int
  :return: The utility values for agents.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.coordination_graph.DCG_payoff(dim_input, dim_hidden, dim_act, args)

  This class defines the payoff function between agents, which is inherits from DCG_utility.

  :param dim_input: The dimension of the input.
  :type dim_input: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param dim_act: The dimension of the actions.
  :type dim_act: int
  :param args: the arguments.
  :type args: Namespace

.. py:function::
  xuance.mindspore.policies.coordination_graph.DCG_payoff.call(hidden_states_n, edges_from, edges_to)

  Computes payoffs based on the provided hidden states and edge information. 
  Supports both low-rank and full-rank payoff computation.

  :param hidden_states_n: The dimension of the hidden states of n agents.
  :type hidden_states_n: int
  :param edges_from: The edges from others to self agent.
  :type edges_from: ms.Tensor
  :param edges_to: The edges from  self agent to others.
  :type edges_to: ms.Tensor
  :return: The provided hidden states and edge information.
  :rtype: ms.Tensor

.. py:class::
  xuance.mindspore.policies.coordination_graph.Coordination_Graph(n_vertexes, graph_type)

  Represents a coordination graph with a specified number of vertices and type.

  :param n_vertexes: The number of vertexes between agents.
  :type n_vertexes: int
  :param graph_type: The type of the topology graph for n agents.
  :type graph_type: str

.. py:function::
  xuance.mindspore.policies.coordination_graph.Coordination_Graph.set_coordination_graph()

  Sets up the coordination graph, including the assignment of edges and related tensors.

.. raw:: html

    <br><hr>


Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

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



  .. group-tab:: TensorFlow

    .. code-block:: python

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


  .. group-tab:: MindSpore

    .. code-block:: python

        import mindspore as ms
        import mindspore.nn as nn
        import torch_scatter
        import torch
        import numpy as np


        class DCG_utility(nn.Cell):
            def __init__(self, dim_input, dim_hidden, dim_output):
                super(DCG_utility, self).__init__()
                self.dim_input = dim_input
                self.dim_hidden = dim_hidden
                self.dim_output = dim_output
                self.output = nn.SequentialCell(nn.Dense(int(self.dim_input), int(self.dim_hidden)),
                                                nn.ReLU(),
                                                nn.Dense(int(self.dim_hidden), int(self.dim_output)))
                # self.output = nn.Sequential(nn.Linear(self.dim_input, self.dim_output))

            def construct(self, hidden_states_n):
                return self.output(hidden_states_n)


        class DCG_payoff(DCG_utility):
            def __init__(self, dim_input, dim_hidden, dim_act, args):
                self.dim_act = dim_act
                self.low_rank_payoff = args.low_rank_payoff
                self.payoff_rank = args.payoff_rank
                dim_payoff_out = 2 * self.payoff_rank * self.dim_act if self.low_rank_payoff else self.dim_act ** 2
                super(DCG_payoff, self).__init__(dim_input, dim_hidden, dim_payoff_out)
                self._concat = ms.ops.Concat(axis=-1)
                self.stack = ms.ops.Stack(axis=0)
                self.expand_dims = ms.ops.ExpandDims()
                self.transpose = ms.ops.Transpose()

            def construct(self, hidden_states_n, edges_from=None, edges_to=None):
                input_payoff = self.stack([self._concat([hidden_states_n[:, edges_from], hidden_states_n[:, edges_to]]),
                                        self._concat([hidden_states_n[:, edges_to], hidden_states_n[:, edges_from]])])
                payoffs = self.output(input_payoff)
                dim = payoffs.shape[0:-1]
                if self.low_rank_payoff:
                    payoffs = payoffs.view(np.prod(dim) * self.payoff_rank, 2, self.dim_act)
                    self.expand_dim(payoffs[:, 1, :], -2)
                    payoffs = ms.ops.matmul(self.expand_dim(payoffs[:, 0, :], -1), self.expand_dim(payoffs[:, 1, :], -2))  # (dim_act * 1) * (1 * dim_act) -> (dim_act * dim_act)
                    payoffs = payoffs.view(tuple(list(dim) + [self.payoff_rank, self.dim_act, self.dim_act])).sum(axis=-3)
                else:
                    payoffs = payoffs.view(tuple(list(dim) + [self.dim_act, self.dim_act]))
                payoffs[1] = self.transpose(payoffs[1], (0, 1, 3, 2))  # f_ij(a_i, a_j) <-> f_ji(a_j, a_i)
                return payoffs.mean(axis=0)  # f^E_{ij} = (f_ij(a_i, a_j) + f_ji(a_j, a_i)) / 2


        class Coordination_Graph(nn.Cell):
            def __init__(self, n_vertexes, graph_type):
                super(Coordination_Graph, self).__init__()
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
                # convert to mindspore tensor
                self.edges_from = ms.Tensor(self.edges_from.numpy())
                self.edges_to = ms.Tensor(self.edges_to.numpy())
                self.edges_n_in = ms.Tensor(self.edges_n_in.numpy())
                return
