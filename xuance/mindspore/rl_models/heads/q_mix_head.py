from gymnasium.spaces import Discrete
from typing import Optional, Dict

from xuance.mindspore import ms, ops, nn, Tensor, Module


class IndependentMixer(Module):
    """
    The independent q-learning. (Independent)
    """

    def construct(self, values_n, states=None):
        return values_n


class VDN_Mixer(Module):
    """
    The value decomposition networks mixer. (Additivity)
    """

    def construct(self, values_n, states=None):
        return ops.reduce_sum(values_n, axis=1)


class QMIX_Mixer(Module):
    """
    The QMIX mixer. (Monotonicity)

    Args:
        dim_state (int): The dimension of global state.
        dim_hidden (int): The size of rach hidden layer.
        dim_hypernet_hidden (int): The size of rach hidden layer for hyper network.
        n_agents (int): The number of agents.
    """

    def __init__(self, dim_state: Optional[int] = None,
                 dim_hidden: int = 32,
                 dim_hypernet_hidden: int = 32,
                 n_agents: int = 1):
        super(QMIX_Mixer, self).__init__()
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
        """
        Returns the total Q-values for multi-agent team.

        Parameters:
            values_n: The individual values for agents in team.
            states: The global states.

        Returns:
            q_tot: The total Q-values for the multi-agent team.
        """

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


class QMIX_FF_Mixer(Module):
    """
    The feedforward mixer without the constraints of monotonicity.
    """

    def __init__(self, dim_state: int = 0,
                 dim_hidden: int = 32,
                 n_agents: int = 1):
        super(QMIX_FF_Mixer, self).__init__()
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
        """
        Returns the feedforward total Q-values.

        Parameters:
            values_n: The individual Q-values.
            states: The global states.
        """

        states = states.reshape(-1, self.dim_state)
        agent_qs = values_n.view(-1, self.n_agents)
        inputs = self._concat([agent_qs, states])
        out_put = self.ff_net(inputs)
        bias = self.ff_net_bias(states)
        y = out_put + bias
        q_tot = y.view(-1, 1)
        return q_tot


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

    def __init__(self, dim_state: int = 0,
                 action_space: Dict[str, Discrete] = None,
                 dim_hidden: int = 32,
                 n_agents: int = 1,
                 dim_utility_hidden: int = 1,
                 use_parameter_sharing: bool = False):
        super(QTRAN_Base, self).__init__()
        self.dim_state = dim_state
        self.action_space = action_space
        self.n_actions_list = [a_space.n for a_space in action_space.values()]
        self.n_actions_max = max(self.n_actions_list)
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.use_parameter_sharing = use_parameter_sharing

        self.dim_q_input = int(self.dim_state + dim_utility_hidden + self.n_actions_max)
        self.dim_v_input = int(self.dim_state)

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
        self.dim_ae_input = int(dim_utility_hidden + self.n_actions_max)
        self.action_encoding = nn.SequentialCell(nn.Dense(self.dim_ae_input, self.dim_ae_input),
                                                 nn.ReLU(),
                                                 nn.Dense(self.dim_ae_input, self.dim_ae_input))

    def construct(self, states: Tensor, hidden_state_inputs: Tensor, actions_onehot: Tensor):
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
        h_state_action_input = ops.cat([hidden_state_inputs, actions_onehot], axis=-1)
        h_state_action_encode = self.action_encoding(h_state_action_input).reshape(-1, self.n_agents, self.dim_ae_input)
        h_state_action_encode = ops.sum(h_state_action_encode, dim=1)  # Sum across agents
        input_q = ops.cat([states, h_state_action_encode], axis=-1)
        input_v = states
        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(input_v)
        return q_jt, v_jt


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

    def __init__(self, dim_state: int = 0,
                 action_space: Dict[str, Discrete] = None,
                 dim_hidden: int = 32,
                 n_agents: int = 1,
                 dim_utility_hidden: int = 1,
                 use_parameter_sharing: bool = False):
        super(QTRAN_Alt, self).__init__()
        self.dim_state = dim_state
        self.action_space = action_space
        self.n_actions_list = [a_space.n for a_space in action_space.values()]
        self.n_actions_max = int(max(self.n_actions_list))
        self.dim_hidden = int(dim_hidden)
        self.n_agents = n_agents
        self.use_parameter_sharing = use_parameter_sharing

        self.dim_q_input = int(self.dim_state + dim_utility_hidden + self.n_actions_max + self.n_agents)
        self.dim_v_input = int(self.dim_state)

        self.Q_jt = nn.SequentialCell(nn.Dense(self.dim_q_input, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, self.n_actions_max))
        self.V_jt = nn.SequentialCell(nn.Dense(self.dim_v_input, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, 1))
        self.dim_ae_input = int(dim_utility_hidden + self.n_actions_max)
        self.action_encoding = nn.SequentialCell(nn.Dense(self.dim_ae_input, self.dim_ae_input),
                                                 nn.ReLU(),
                                                 nn.Dense(self.dim_ae_input, self.dim_ae_input))

    def construct(self, states: Tensor, hidden_state_inputs: Tensor, actions_onehot: Tensor):
        """Calculating the joint Q and V values.

        Parameters:
            states (Tensor): The global states.
            hidden_state_inputs (Tensor): The joint hidden states inputs for QTRAN network.
            actions_onehot (Tensor): The joint onehot actions for QTRAN network.

        Returns:
            q_jt (Tensor): The evaluated joint Q values.
            v_jt (Tensor): The evaluated joint V values.
        """
        h_state_action_input = ops.cat([hidden_state_inputs, actions_onehot], axis=-1)
        h_state_action_encode = self.action_encoding(h_state_action_input).reshape(-1, self.n_agents, self.dim_ae_input)
        bs, dim_h = h_state_action_encode.shape[0], h_state_action_encode.shape[-1]
        agent_ids = ops.eye(self.n_agents, dtype=ms.float32)
        agent_masks = (1 - agent_ids)
        repeat_agent_ids = ops.repeat_elements(agent_ids.unsqueeze(0), rep=bs, axis=0)
        repeated_agent_masks = ops.repeat_elements(agent_masks.unsqueeze(0).unsqueeze(-1), rep=bs, axis=0)
        repeated_agent_masks = ops.repeat_elements(repeated_agent_masks, rep=dim_h, axis=-1)
        repeated_h_state_action_encode = ops.repeat_elements(h_state_action_encode.unsqueeze(2),
                                                             rep=self.n_agents, axis=2)
        h_state_action_encode = repeated_h_state_action_encode * repeated_agent_masks
        h_state_action_encode = h_state_action_encode.sum(axis=2)  # Sum across other agents

        repeated_states = ops.repeat_elements(states.unsqueeze(1), rep=self.n_agents, axis=1)
        input_q = ops.cat([repeated_states, h_state_action_encode, repeat_agent_ids], axis=-1)
        input_v = states
        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(input_v)
        return q_jt, v_jt
