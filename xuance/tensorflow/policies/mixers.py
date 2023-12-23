import tensorflow as tf
import tensorflow.keras as tk


class VDN_mixer(tk.Model):
    def __init__(self):
        super(VDN_mixer, self).__init__()

    def call(self, values_n, states=None, **kwargs):
        return tf.reduce_sum(values_n, axis=1)


class QMIX_mixer(tk.Model):
    def __init__(self, dim_state, dim_hidden, dim_hypernet_hidden, n_agents, device):
        super(QMIX_mixer, self).__init__()
        self.device = device
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

    def call(self, values_n, states=None, **kwargs):
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

    def call(self, values_n, states=None, **kwargs):
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

    def call(self, hidden_states_n, actions_n=None, **kwargs):
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
