from gymnasium.spaces import Discrete
from typing import Optional, Dict
from xuance.tensorflow import tf, Tensor, Module


class IndependentMixer(Module):
    """
    The independent q-learning. (Independent)
    """

    def call(self, values_n, states=None):
        return values_n


class VDN_Mixer(Module):
    """
    The value decomposition networks mixer. (Additivity)
    """

    def call(self, values_n, states=None):
        return tf.reduce_sum(values_n, axis=-1, keepdims=True)


class QMIX_Mixer(Module):
    """
    The QMIX mixer. (Monotonicity)

    Args:
        dim_state (int): The dimension of global state.
        dim_hidden (int): The size of rach hidden layer.
        dim_hypernet_hidden (int): The size of rach hidden layer for hyper network.
        n_agents (int): The number of agents.
    """

    def __init__(
            self,
            dim_state: Optional[int] = None,
            dim_hidden: int = 32,
            dim_hypernet_hidden: int = 32,
            n_agents: int = 1,
    ):
        super(QMIX_Mixer, self).__init__()
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_hypernet_hidden = dim_hypernet_hidden
        self.n_agents = n_agents

        self.hyper_w_1 = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dim_hypernet_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.dim_hidden * self.n_agents),
        ])

        self.hyper_w_2 = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dim_hypernet_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.dim_hidden),
        ])

        self.hyper_b_1 = tf.keras.layers.Dense(self.dim_hidden)

        self.hyper_b_2 = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dim_hypernet_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1),
        ])

    def call(self, values_n, states):
        """
        Returns the total Q-values for multi-agent team.

        Parameters:
            values_n: The individual values for agents in team.
            states: The global states.

        Returns:
            q_tot: The total Q-values for the multi-agent team.
        """
        states = tf.cast(states, tf.float32)
        states = tf.reshape(states, (-1, self.dim_state))
        agent_qs = tf.reshape(values_n, (-1, 1, self.n_agents))

        # First layer
        w_1 = tf.abs(self.hyper_w_1(states))
        w_1 = tf.reshape(w_1, (-1, self.n_agents, self.dim_hidden))

        b_1 = self.hyper_b_1(states)
        b_1 = tf.reshape(b_1, (-1, 1, self.dim_hidden))

        hidden = tf.nn.elu(tf.matmul(agent_qs, w_1) + b_1)

        # Second layer
        w_2 = tf.abs(self.hyper_w_2(states))
        w_2 = tf.reshape(w_2, (-1, self.dim_hidden, 1))

        b_2 = self.hyper_b_2(states)
        b_2 = tf.reshape(b_2, (-1, 1, 1))

        # Compute final output
        y = tf.matmul(hidden, w_2) + b_2
        # Reshape and return
        q_tot = tf.reshape(y, (-1, 1))
        return q_tot

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            dim_state=self.dim_state,
            dim_hidden=self.dim_hidden,
            dim_hypernet_hidden=self.dim_hypernet_hidden,
            n_agents=self.n_agents,
        ))
        return config


class QMIX_FF_Mixer(Module):
    """
    The feedforward mixer without the constraints of monotonicity.
    """

    def __init__(
            self,
            dim_state: int = 0,
            dim_hidden: int = 32,
            n_agents: int = 1,
    ):
        super(QMIX_FF_Mixer, self).__init__()
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_input = self.n_agents + self.dim_state

        self.ff_net = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dim_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.dim_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.dim_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1),
        ])
        self.ff_net_bias = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dim_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1),
        ])

    def call(self, values_n, states=None):
        """
        Returns the feedforward total Q-values.

        Parameters:
            values_n: The individual Q-values.
            states: The global states.
        """
        states = tf.cast(states, tf.float32)
        states = tf.reshape(states, (-1, self.dim_state))
        agent_qs = tf.reshape(values_n, (-1, self.n_agents))

        inputs = tf.concat([agent_qs, states], axis=-1)

        out_put = self.ff_net(inputs)
        bias = self.ff_net_bias(states)

        y = out_put + bias
        q_tot = tf.reshape(y, (-1, 1))

        return q_tot

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            dim_state=self.dim_state,
            dim_hidden=self.dim_hidden,
            n_agents=self.n_agents,
        ))


class QTRAN_Base(Module):
    """
    The basic QTRAN module.

    Args:
        dim_state (int): The dimension of the global state.
        action_space (Dict[str, Discrete]): The action space for all agents.
        dim_hidden (int): The dimension of the hidden layers.
        n_agents (int): The number of agents.
        dim_utility_hidden (int): The dimension of the utility hidden states.
        use_parameter_sharing (bool): Whether to use parameters sharing trick.
    """

    def __init__(
            self,
            dim_state: int = 0,
            action_space: Dict[str, Discrete] = None,
            dim_hidden: int = 32,
            n_agents: int = 1,
            dim_utility_hidden: int = 1,
            use_parameter_sharing: bool = False,
    ):
        super(QTRAN_Base, self).__init__()
        self.dim_state = dim_state
        self.action_space = action_space
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_utility_hidden = dim_utility_hidden
        self.use_parameter_sharing = use_parameter_sharing

        self.n_actions_list = [a_space.n for a_space in action_space.values()]
        self.n_actions_max = max(self.n_actions_list)

        self.dim_q_input = self.dim_state + dim_utility_hidden + self.n_actions_max
        self.dim_v_input = self.dim_state

        self.Q_jt = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dim_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.dim_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1),
        ])

        self.V_jt = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dim_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.dim_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1),
        ])

        self.dim_ae_input = dim_utility_hidden + self.n_actions_max
        self.action_encoding = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dim_ae_input),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.dim_ae_input),
        ])

    def call(self, states: Tensor, hidden_state_inputs: Tensor, actions_onehot: Tensor):
        """
        Calculating the joint Q and V values.

        Parameters:
            states (Tensor): The global states.
            hidden_state_inputs (Tensor): The joint hidden states inputs for QTRAN network.
            actions_onehot (Tensor): The joint onehot actions for QTRAN network.

        Returns:
            q_jt (Tensor): The evaluated joint Q values.
            v_jt (Tensor): The evaluated joint V values.
        """
        h_state_action_input = tf.concat([hidden_state_inputs, actions_onehot], axis=-1)
        h_state_action_encode = self.action_encoding(h_state_action_input)
        h_state_action_encode = tf.reshape(h_state_action_encode, (-1, self.n_agents, self.dim_ae_input))

        # Sum across agents.
        h_state_action_encode = tf.reduce_sum(h_state_action_encode, axis=1)
        states = tf.cast(states, tf.float32)
        input_q = tf.concat([states, h_state_action_encode], axis=-1, )
        input_v = states

        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(input_v)

        return q_jt, v_jt

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            dim_state=self.dim_state,
            action_space=self.action_space,
            dim_hidden=self.dim_hidden,
            n_agents=self.n_agents,
            dim_utility_hidden=self.dim_utility_hidden,
            use_parameter_sharing=self.use_parameter_sharing,
        ))
        return config


class QTRAN_Alt(Module):
    """
    The basic QTRAN module.

    Parameters:
        dim_state (int): The dimension of the global state.
        action_space (Dict[str, Discrete]): The action space for all agents.
        dim_hidden (int): The dimension of the hidden layers.
        n_agents (int): The number of agents.
        dim_utility_hidden (int): The dimension of the utility hidden states.
        use_parameter_sharing (bool): Whether to use parameters sharing trick.
    """

    def __init__(
            self,
            dim_state: int = 0,
            action_space: Dict[str, Discrete] = None,
            dim_hidden: int = 32,
            n_agents: int = 1,
            dim_utility_hidden: int = 1,
            use_parameter_sharing: bool = False,
    ):
        super(QTRAN_Alt, self).__init__()
        self.dim_state = dim_state
        self.action_space = action_space
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_utility_hidden = dim_utility_hidden
        self.use_parameter_sharing = use_parameter_sharing

        self.n_actions_list = [a_space.n for a_space in action_space.values()]
        self.n_actions_max = max(self.n_actions_list)

        self.dim_q_input = self.dim_state + dim_utility_hidden + self.n_actions_max + self.n_agents
        self.dim_v_input = self.dim_state

        self.Q_jt = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dim_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.dim_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.n_actions_max),
        ])

        self.V_jt = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dim_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.dim_hidden),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1),
        ])

        self.dim_ae_input = dim_utility_hidden + self.n_actions_max

        self.action_encoding = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dim_ae_input),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.dim_ae_input),
        ])

    def call(self, states: Tensor, hidden_state_inputs: Tensor, actions_onehot: Tensor):
        """Calculating the joint Q and V values.

        Parameters:
            states (Tensor): The global states.
            hidden_state_inputs (Tensor): The joint hidden states inputs for QTRAN network.
            actions_onehot (Tensor): The joint onehot actions for QTRAN network.

        Returns:
            q_jt (Tensor): The evaluated joint Q values.
            v_jt (Tensor): The evaluated joint V values.
        """
        h_state_action_input = tf.concat([hidden_state_inputs, actions_onehot], axis=-1)
        h_state_action_encode = self.action_encoding(h_state_action_input)
        h_state_action_encode = tf.reshape(h_state_action_encode, (-1, self.n_agents, self.dim_ae_input))
        bs, dim_h = tf.shape(h_state_action_encode)[0], tf.shape(h_state_action_encode)[-1]
        agent_ids = tf.eye(self.n_agents, dtype=tf.float32)
        agent_masks = 1.0 - agent_ids
        repeat_agent_ids = tf.tile(tf.expand_dims(agent_ids, axis=0), [bs, 1, 1], )
        repeated_agent_masks = tf.tile(tf.expand_dims(tf.expand_dims(agent_masks, axis=0), axis=-1, ),
                                       [bs, 1, 1, self.dim_ae_input])
        repeated_h_state_action_encode = tf.tile(tf.expand_dims(h_state_action_encode, axis=2),
                                                 [1, 1, self.n_agents, 1])
        h_state_action_encode = repeated_h_state_action_encode * repeated_agent_masks
        h_state_action_encode = tf.reduce_sum(h_state_action_encode, axis=2)  # Sum across other agents

        states = tf.cast(states, tf.float32)

        repeated_states = tf.tile(tf.expand_dims(states, axis=1), [1, self.n_agents, 1])

        input_q = tf.concat([repeated_states, h_state_action_encode, repeat_agent_ids, ], axis=-1)
        input_v = states
        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(input_v)
        return q_jt, v_jt

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            dim_state=self.dim_state,
            action_space=self.action_space,
            dim_hidden=self.dim_hidden,
            n_agents=self.n_agents,
            dim_utility_hidden=self.dim_utility_hidden,
            use_parameter_sharing=self.use_parameter_sharing,
        ))
        return config

