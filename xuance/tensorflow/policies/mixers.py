import tensorflow as tf
import tensorflow.keras as tk
import numpy as np
import torch
import torch_scatter


class VDN_mixer(tk.Model):
    def __init__(self):
        super(VDN_mixer, self).__init__()

    def call(self, values_n, states=None, training=None, masks=None):
        return tf.reduce_sum(values_n, axis=1)


class QMIX_mixer(tk.Model):
    def __init__(self, dim_state, dim_hidden, dim_hypernet_hidden, n_agents):
        super(QMIX_mixer, self).__init__()
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_hypernet_hidden = dim_hypernet_hidden
        self.n_agents = n_agents
        # self.hyper_w_1 = nn.Linear(self.dim_state, self.dim_hidden * self.n_agents)
        # self.hyper_w_2 = nn.Linear(self.dim_state, self.dim_hidden)
        linear_w_1 = [tk.layers.Dense(units=self.dim_hypernet_hidden,
                                      activation=tk.layers.Activation('relu'),
                                      input_shape=(self.dim_state,)),
                      tk.layers.Dense(units=self.dim_hidden * self.n_agents, input_shape=(self.dim_hypernet_hidden,))]
        self.hyper_w_1 = tk.Sequential(linear_w_1)
        linear_w_2 = [tk.layers.Dense(units=self.dim_hypernet_hidden,
                                      activation=tk.layers.Activation('relu'),
                                      input_shape=(self.dim_state,)),
                      tk.layers.Dense(units=self.dim_hidden, input_shape=(self.dim_hypernet_hidden,))]
        self.hyper_w_2 = tk.Sequential(linear_w_2)

        self.hyper_b_1 = tk.layers.Dense(units=self.dim_hidden, input_shape=(self.dim_state,))
        self.hyper_b_2 = tk.Sequential([tk.layers.Dense(units=self.dim_hypernet_hidden,
                                                        activation=tk.layers.Activation('relu'),
                                                        input_shape=(self.dim_state,)),
                                        tk.layers.Dense(units=1, input_shape=(self.dim_hypernet_hidden,))])

    def call(self, values_n, states=None, training=None, masks=None):
        states = tf.reshape(states, [-1, self.dim_state])
        agent_qs = tf.reshape(values_n, [-1, 1, self.n_agents])
        # First layer
        w_1 = tf.abs(self.hyper_w_1(states))
        w_1 = tf.reshape(w_1, [-1, self.n_agents, self.dim_hidden])
        b_1 = self.hyper_b_1(states)
        b_1 = tf.reshape(b_1, [-1, 1, self.dim_hidden])
        hidden = tf.nn.elu(tf.linalg.matmul(agent_qs, w_1) + b_1)
        # Second layer
        w_2 = tf.abs(self.hyper_w_2(states))
        w_2 = tf.reshape(w_2, [-1, self.dim_hidden, 1])
        b_2 = self.hyper_b_2(states)
        b_2 = tf.reshape(b_2, [-1, 1, 1])
        # Compute final output
        y = tf.linalg.matmul(hidden, w_2) + b_2
        # Reshape and return
        q_tot = tf.reshape(y, [-1, 1])
        return q_tot


class QMIX_FF_mixer(tk.Model):
    def __init__(self, dim_state, dim_hidden, n_agents):
        super(QMIX_FF_mixer, self).__init__()
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_input = self.n_agents + self.dim_state
        tk.layers.Dense(input_shape=(self.dim_input,), units=self.dim_hidden, activation=tk.layers.Activation('relu'))
        layers_ff_net = [tk.layers.Dense(input_shape=(self.dim_input,), units=self.dim_hidden,
                                         activation=tk.layers.Activation('relu')),
                         tk.layers.Dense(input_shape=(self.dim_hidden,), units=self.dim_hidden,
                                         activation=tk.layers.Activation('relu')),
                         tk.layers.Dense(input_shape=(self.dim_hidden,), units=self.dim_hidden,
                                         activation=tk.layers.Activation('relu')),
                         tk.layers.Dense(input_shape=(self.dim_hidden,), units=1)]
        self.ff_net = tk.Sequential(layers_ff_net)
        layers_ff_net_bias = [tk.layers.Dense(input_shape=(self.dim_state,), units=self.dim_hidden,
                                              activation=tk.layers.Activation('relu')),
                              tk.layers.Dense(input_shape=(self.dim_hidden,), units=1)]
        self.ff_net_bias = tk.Sequential(layers_ff_net_bias)

    def call(self, values_n, states=None, training=None, masks=None):
        states = tf.reshape(states, [-1, self.dim_state])
        agent_qs = tf.reshape(values_n, [-1, self.n_agents])
        inputs = tf.concat([agent_qs, states], axis=-1)
        out_put = self.ff_net(inputs)
        bias = self.ff_net_bias(states)
        y = out_put + bias
        q_tot = tf.reshape(y, [-1, 1])
        return q_tot


class QTRAN_base(tk.Model):
    def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
        super(QTRAN_base, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_q_input = (dim_utility_hidden + self.dim_action) * self.n_agents
        self.dim_v_input = dim_utility_hidden * self.n_agents

        linear_Q_jt = [tk.layers.Dense(input_shape=(self.dim_q_input,), units=self.dim_hidden,
                                       activation=tk.layers.Activation('relu')),
                       tk.layers.Dense(input_shape=(self.dim_hidden,), units=self.dim_hidden,
                                       activation=tk.layers.Activation('relu')),
                       tk.layers.Dense(input_shape=(self.dim_hidden,), units=1)]
        self.Q_jt = tk.Sequential(linear_Q_jt)
        linear_V_jt = [tk.layers.Dense(input_shape=(self.dim_v_input,), units=self.dim_hidden,
                                       activation=tk.layers.Activation('relu')),
                       tk.layers.Dense(input_shape=(self.dim_hidden,), units=self.dim_hidden,
                                       activation=tk.layers.Activation('relu')),
                       tk.layers.Dense(input_shape=(self.dim_hidden,), units=1)]
        self.V_jt = tk.Sequential(linear_V_jt)

    def call(self, hidden_states_n, actions_n=None, training=None, masks=None):
        input_q = tf.reshape(tf.concat([hidden_states_n, actions_n], axis=-1), [-1, self.dim_q_input])
        input_v = tf.reshape(hidden_states_n, [-1, self.dim_v_input])
        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(input_v)
        return q_jt, v_jt


class QTRAN_alt(QTRAN_base):
    def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
        super(QTRAN_alt, self).__init__(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

    def counterfactual_values(self, q_self_values, q_selected_values):
        q_repeat = tf.tile(tf.expand_dims(q_selected_values, axis=1), multiples=(1, self.n_agents, 1, self.dim_action))
        counterfactual_values_n = q_repeat.numpy()
        for agent in range(self.n_agents):
            counterfactual_values_n[:, agent, agent] = q_self_values[:, agent, :].numpy()
        counterfactual_values_n = tf.convert_to_tensor(counterfactual_values_n)
        return tf.reduce_sum(counterfactual_values_n, axis=2)

    def counterfactual_values_hat(self, hidden_states_n, actions_n):
        action_repeat = tf.tile(tf.expand_dims(actions_n, axis=2), multiples=(1, 1, self.dim_action, 1))
        action_self_all = tf.expand_dims(tf.eye(self.dim_action), axis=0).numpy()
        action_counterfactual_n = tf.tile(tf.expand_dims(action_repeat, axis=2), multiples=(
            1, 1, self.n_agents, 1, 1)).numpy()  # batch * N * N * dim_a * dim_a
        q_n = []
        for agent in range(self.n_agents):
            action_counterfactual_n[:, agent, agent, :, :] = action_self_all
            q_actions = []
            for a in range(self.dim_action):
                input_a = tf.convert_to_tensor(action_counterfactual_n[:, :, agent, a, :])
                q, _ = self.call(hidden_states_n, input_a)
                q_actions.append(q)
            q_n.append(tf.expand_dims(tf.concat(q_actions, axis=-1), axis=1))
        return tf.concat(q_n, axis=1)


class DCG_utility(tk.Model):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super(DCG_utility, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        linears_layers = [tk.layers.Dense(input_shape=(self.dim_input,), units=self.dim_hidden,
                                          activation=tk.layers.Activation('relu')),
                          tk.layers.Dense(input_shape=(self.dim_input,), units=self.dim_output)]

        self.model = tk.Sequential(linears_layers)
        # self.output = tk.Sequential(nn.Linear(self.dim_input, self.dim_output))

    def call(self, hidden_states_n, training=None, masks=None):
        return self.model(hidden_states_n)


class DCG_payoff(DCG_utility):
    def __init__(self, dim_input, dim_hidden, dim_act, args):
        self.dim_act = dim_act
        self.low_rank_payoff = args.low_rank_payoff
        self.payoff_rank = args.payoff_rank
        dim_payoff_out = 2 * self.payoff_rank * self.dim_act if self.low_rank_payoff else self.dim_act ** 2
        super(DCG_payoff, self).__init__(dim_input, dim_hidden, dim_payoff_out)
        self.input_payoff_shape = None

    def call(self, hidden_from_to, hidden_to_from=None, training=None, masks=None):
        # input_payoff = tf.stack([tf.convert_to_tensor(hidden_from_to), tf.convert_to_tensor(hidden_to_from)], axis=0)
        input_payoff = tf.stack([hidden_from_to, hidden_to_from], axis=0)

        self.input_payoff_shape = input_payoff.shape
        return self.model(tf.reshape(input_payoff, [-1, self.input_payoff_shape[-1]]))

    def mean_payoffs(self, payoffs):
        payoffs = tf.reshape(payoffs, self.input_payoff_shape[0:-1] + (self.dim_output,))
        dim = payoffs.shape[0:-1]
        if self.low_rank_payoff:
            payoffs = tf.reshape(payoffs, [np.prod(dim) * self.payoff_rank, 2, self.dim_act])
            payoffs_0 = tf.convert_to_tensor(payoffs.numpy()[:, 0, :])
            payoffs_1 = tf.convert_to_tensor(payoffs.numpy()[:, 1, :])
            payoffs = tf.matmul(tf.expand_dims(payoffs_0, axis=-1), tf.expand_dims(payoffs_1,
                                                                                   axis=-2))  # (dim_act * 1) * (1 * dim_act) -> (dim_act * dim_act)
            payoffs = tf.reshape(payoffs, list(dim) + [self.payoff_rank, self.dim_act, self.dim_act])
            payoffs = tf.reduce_sum(payoffs, axis=-3)
        else:
            payoffs = tf.reshape(payoffs, list(dim) + [self.dim_act, self.dim_act])

        payoffs = payoffs.numpy()
        dim_num = len(payoffs.shape) - 1
        dim_trans = list(np.arange(dim_num - 2)) + [dim_num - 1, dim_num - 2]
        payoffs[1] = np.transpose(payoffs[1], dim_trans).copy()
        return payoffs.mean(axis=0)  # f^E_{ij} = (f_ij(a_i, a_j) + f_ji(a_j, a_i)) / 2


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
        self.edges_from = torch.zeros(self.n_edges).long()
        self.edges_to = torch.zeros(self.n_edges).long()
        for i, edge in enumerate(self.edges):
            self.edges_from[i] = edge[0]
            self.edges_to[i] = edge[1]
        self.edges_n_in = torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                    index=self.edges_to, dim=0, dim_size=self.n_vertexes) \
                          + torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                      index=self.edges_from, dim=0, dim_size=self.n_vertexes)
        self.edges_n_in = self.edges_n_in.float().numpy()
        self.edges_from = self.edges_from.numpy()
        self.edges_to = self.edges_to.numpy()
        return
