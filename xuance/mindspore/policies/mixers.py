import mindspore as ms
import mindspore.nn as nn
import torch_scatter
import torch
import numpy as np


class VDN_mixer(nn.Cell):
    def __init__(self):
        super(VDN_mixer, self).__init__()
        self._sum = ms.ops.ReduceSum(keep_dims=False)

    def construct(self, values_n, states=None):
        return self._sum(values_n, 1)


class QMIX_mixer(nn.Cell):
    def __init__(self, dim_state, dim_hidden, dim_hypernet_hidden, n_agents):
        super(QMIX_mixer, self).__init__()
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_hypernet_hidden = dim_hypernet_hidden
        self.n_agents = n_agents
        # self.hyper_w_1 = nn.Linear(self.dim_state, self.dim_hidden * self.n_agents)
        # self.hyper_w_2 = nn.Linear(self.dim_state, self.dim_hidden)
        self.hyper_w_1 = nn.SequentialCell(nn.Dense(self.dim_state, self.dim_hypernet_hidden),
                                           nn.ReLU(),
                                           nn.Dense(self.dim_hypernet_hidden, self.dim_hidden * self.n_agents))
        self.hyper_w_2 = nn.SequentialCell(nn.Dense(self.dim_state, self.dim_hypernet_hidden),
                                           nn.ReLU(),
                                           nn.Dense(self.dim_hypernet_hidden, self.dim_hidden))

        self.hyper_b_1 = nn.Dense(self.dim_state, self.dim_hidden)
        self.hyper_b_2 = nn.SequentialCell(nn.Dense(self.dim_state, self.dim_hypernet_hidden),
                                           nn.ReLU(),
                                           nn.Dense(self.dim_hypernet_hidden, 1))
        self._abs = ms.ops.Abs()
        self._elu = ms.ops.Elu()

    def construct(self, values_n, states):
        states = states.reshape(-1, self.dim_state)
        agent_qs = values_n.view(-1, 1, self.n_agents)
        # First layer
        w_1 = self._abs(self.hyper_w_1(states))
        w_1 = w_1.view(-1, self.n_agents, self.dim_hidden)
        b_1 = self.hyper_b_1(states)
        b_1 = b_1.view(-1, 1, self.dim_hidden)
        hidden = self._elu(ms.ops.matmul(agent_qs, w_1) + b_1)
        # Second layer
        w_2 = self._abs(self.hyper_w_2(states))
        w_2 = w_2.view(-1, self.dim_hidden, 1)
        b_2 = self.hyper_b_2(states)
        b_2 = b_2.view(-1, 1, 1)
        # Compute final output
        y = ms.ops.matmul(hidden, w_2) + b_2
        # Reshape and return
        q_tot = y.view(-1, 1)
        return q_tot


class QMIX_FF_mixer(nn.Cell):
    def __init__(self, dim_state, dim_hidden, n_agents):
        super(QMIX_FF_mixer, self).__init__()
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_input = self.n_agents + self.dim_state
        self.ff_net = nn.SequentialCell(nn.Dense(self.dim_input, self.dim_hidden),
                                        nn.ReLU(),
                                        nn.Dense(self.dim_hidden, self.dim_hidden),
                                        nn.ReLU(),
                                        nn.Dense(self.dim_hidden, self.dim_hidden),
                                        nn.ReLU(),
                                        nn.Dense(self.dim_hidden, 1))
        self.ff_net_bias = nn.SequentialCell(nn.Dense(self.dim_state, self.dim_hidden),
                                             nn.ReLU(),
                                             nn.Dense(self.dim_hidden, 1))
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, values_n, states):
        states = states.reshape(-1, self.dim_state)
        agent_qs = values_n.view(-1, self.n_agents)
        inputs = self._concat([agent_qs, states])
        out_put = self.ff_net(inputs)
        bias = self.ff_net_bias(states)
        y = out_put + bias
        q_tot = y.view(-1, 1)
        return q_tot


class QTRAN_base(nn.Cell):
    def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
        super(QTRAN_base, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_q_input = (dim_utility_hidden + self.dim_action) * self.n_agents
        self.dim_v_input = dim_utility_hidden * self.n_agents

        self.Q_jt = nn.SequentialCell(nn.Dense(self.dim_q_input, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, 1))
        self.V_jt = nn.SequentialCell(nn.Dense(self.dim_v_input, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, 1))
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, hidden_states_n, actions_n):
        input_q = self._concat([hidden_states_n, actions_n]).view(-1, self.dim_q_input)
        input_v = hidden_states_n.view(-1, self.dim_v_input)
        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(input_v)
        return q_jt, v_jt


class QTRAN_alt(QTRAN_base):
    def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
        super(QTRAN_alt, self).__init__(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

    def counterfactual_values(self, q_self_values, q_selected_values):
        q_repeat = ms.ops.broadcast_to(ms.ops.expand_dims(q_selected_values, axis=1),
                                       (-1, self.n_agents, -1, self.dim_action))
        counterfactual_values_n = q_repeat
        for agent in range(self.n_agents):
            counterfactual_values_n[:, agent, agent] = q_self_values[:, agent, :]
        return counterfactual_values_n.sum(axis=2)

    def counterfactual_values_hat(self, hidden_states_n, actions_n):
        action_repeat = ms.ops.broadcast_to(ms.ops.expand_dims(actions_n, axis=2), (-1, -1, self.dim_action, -1))
        action_self_all = ms.ops.expand_dims(ms.ops.eye(self.dim_action, self.dim_action, ms.float32), axis=0)
        action_counterfactual_n = ms.ops.broadcast_to(ms.ops.expand_dims(action_repeat, axis=2),
                                                      (-1, -1, self.n_agents, -1, -1))  # batch * N * N * dim_a * dim_a

        q_n = []
        for agent in range(self.n_agents):
            action_counterfactual_n[:, agent, agent, :, :] = action_self_all
            q_actions = []
            for a in range(self.dim_action):
                input_a = action_counterfactual_n[:, :, agent, a, :]
                q, _ = self.construct(hidden_states_n, input_a)
                q_actions.append(q)
            q_n.append(ms.ops.expand_dims(self._concat(q_actions), axis=1))
        return ms.ops.concat(q_n, axis=1)


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
